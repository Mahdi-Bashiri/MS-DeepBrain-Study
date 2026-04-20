###################### Libraries ######################
# Deep Learning
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras import backend as K
from tensorflow.keras import layers, optimizers, callbacks
from keras.utils import to_categorical


def build_deeplabv3_unet_3class(input_shape=(256, 256, 1), num_classes=3):
    """
    DeepLabV3+ with ResNet-50 backbone.
    
    Key fix over the original:
      - All BatchNormalization replaced with GroupNormalization (groups=8).
        GroupNorm is batch-size independent, so inference statistics are
        identical whether training=True or training=False — no more need to
        force training=True at inference time.
    
    Input:  single-channel (grayscale) MRI images  →  (H, W, 1)
    Output: per-pixel class probabilities           →  (H, W, num_classes)
            or binary mask                          →  (H, W, 1)  when num_classes==1
    
    Reference:
        "Encoder-Decoder with Atrous Separable Convolution for
         Semantic Image Segmentation", Chen et al. 2018.
    """

    # ------------------------------------------------------------------
    # Helper: GroupNorm drop-in for BatchNorm
    # groups=8 works well for filter counts ≥ 32 that are multiples of 8.
    # ------------------------------------------------------------------
    def group_norm(name=None):
        return layers.GroupNormalization(groups=4, name=name)

    # ------------------------------------------------------------------
    def conv_block(x, filters, kernel_size=3, strides=1,
                   dilation_rate=1, use_bias=False, name=None):
        """Standard convolution block with GroupNorm and ReLU."""
        x = layers.Conv2D(
            filters, kernel_size, strides=strides, padding='same',
            dilation_rate=dilation_rate, use_bias=use_bias, name=name
        )(x)
        x = group_norm()(x)
        x = layers.Activation('relu')(x)
        return x

    # ------------------------------------------------------------------
    def bottleneck_residual_block(x, filters, strides=1, dilation_rate=1,
                                  projection_shortcut=False, name_prefix=""):
        """ResNet-50 bottleneck block with optional atrous convolution."""
        shortcut = x

        # Projection shortcut if dimensions change
        if projection_shortcut:
            shortcut = layers.Conv2D(
                filters * 4, 1, strides=strides, use_bias=False,
                name=f"{name_prefix}_0_conv"
            )(shortcut)
            shortcut = group_norm(name=f"{name_prefix}_0_gn")(shortcut)

        # 1×1 → 3×3 (possibly atrous) → 1×1  bottleneck
        x = layers.Conv2D(filters, 1, use_bias=False,
                          name=f"{name_prefix}_1_conv")(x)
        x = group_norm(name=f"{name_prefix}_1_gn")(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(
            filters, 3, strides=strides, padding='same',
            dilation_rate=dilation_rate, use_bias=False,
            name=f"{name_prefix}_2_conv"
        )(x)
        x = group_norm(name=f"{name_prefix}_2_gn")(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters * 4, 1, use_bias=False,
                          name=f"{name_prefix}_3_conv")(x)
        x = group_norm(name=f"{name_prefix}_3_gn")(x)

        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        return x

    # ------------------------------------------------------------------
    def aspp_block(x, filters=256):
        """Atrous Spatial Pyramid Pooling."""

        # Branch 1 — 1×1 conv
        b1 = layers.Conv2D(filters, 1, use_bias=False, name='aspp_1x1')(x)
        b1 = group_norm(name='aspp_1x1_gn')(b1)
        b1 = layers.Activation('relu')(b1)

        # Branch 2 — 3×3, rate=6
        b2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=6,
                           use_bias=False, name='aspp_3x3_6')(x)
        b2 = group_norm(name='aspp_3x3_6_gn')(b2)
        b2 = layers.Activation('relu')(b2)

        # Branch 3 — 3×3, rate=12
        b3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=12,
                           use_bias=False, name='aspp_3x3_12')(x)
        b3 = group_norm(name='aspp_3x3_12_gn')(b3)
        b3 = layers.Activation('relu')(b3)

        # Branch 4 — 3×3, rate=18
        b4 = layers.Conv2D(filters, 3, padding='same', dilation_rate=18,
                           use_bias=False, name='aspp_3x3_18')(x)
        b4 = group_norm(name='aspp_3x3_18_gn')(b4)
        b4 = layers.Activation('relu')(b4)

        # Branch 5 — image-level global context via GAP + resize
        input_shape_dyn = tf.shape(x)
        h, w = input_shape_dyn[1], input_shape_dyn[2]

        b5 = layers.GlobalAveragePooling2D(name='aspp_gap')(x)
        b5 = layers.Reshape((1, 1, -1))(b5)
        b5 = layers.Conv2D(filters, 1, use_bias=False,
                           name='aspp_gap_conv')(b5)
        b5 = group_norm(name='aspp_gap_gn')(b5)
        b5 = layers.Activation('relu')(b5)
        b5 = layers.Lambda(
            lambda args: tf.image.resize(args[0], args[1], method='bilinear'),
            name='aspp_gap_resize'
        )([b5, [h, w]])

        # Fuse all branches
        concat = layers.Concatenate(name='aspp_concat')([b1, b2, b3, b4, b5])
        out = layers.Conv2D(filters, 1, use_bias=False,
                            name='aspp_final_conv')(concat)
        out = group_norm(name='aspp_final_gn')(out)
        out = layers.Activation('relu')(out)
        out = layers.Dropout(0.1, name='aspp_dropout')(out)
        return out

    # ==================================================================
    # INPUT — grayscale, single channel
    # ==================================================================
    inputs = layers.Input(input_shape, name='input')   # (H, W, 1)

    # ==================================================================
    # ENCODER — ResNet-50 backbone
    # ==================================================================

    # Stem
    x = layers.Conv2D(64, 7, strides=2, padding='same',
                      use_bias=False, name='conv1')(inputs)
    x = group_norm(name='conv1_gn')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)

    # Stage 1 — conv2_x  (output stride 4 → low-level features for decoder)
    x = bottleneck_residual_block(x, 64, strides=1,
                                  projection_shortcut=True,
                                  name_prefix='conv2_block1')
    x = bottleneck_residual_block(x, 64, name_prefix='conv2_block2')
    low_level_features = bottleneck_residual_block(x, 64,
                                                   name_prefix='conv2_block3')

    # Stage 2 — conv3_x  (output stride 8)
    x = bottleneck_residual_block(low_level_features, 128, strides=2,
                                  projection_shortcut=True,
                                  name_prefix='conv3_block1')
    x = bottleneck_residual_block(x, 128, name_prefix='conv3_block2')
    x = bottleneck_residual_block(x, 128, name_prefix='conv3_block3')
    x = bottleneck_residual_block(x, 128, name_prefix='conv3_block4')

    # Stage 3 — conv4_x  (atrous rate=2, keeps stride at 8)
    x = bottleneck_residual_block(x, 256, strides=1, dilation_rate=2,
                                  projection_shortcut=True,
                                  name_prefix='conv4_block1')
    for i in range(2, 7):
        x = bottleneck_residual_block(x, 256, dilation_rate=2,
                                      name_prefix=f'conv4_block{i}')

    # Stage 4 — conv5_x  (atrous rate=4, keeps stride at 8)
    x = bottleneck_residual_block(x, 512, strides=1, dilation_rate=4,
                                  projection_shortcut=True,
                                  name_prefix='conv5_block1')
    x = bottleneck_residual_block(x, 512, dilation_rate=4,
                                  name_prefix='conv5_block2')
    x = bottleneck_residual_block(x, 512, dilation_rate=4,
                                  name_prefix='conv5_block3')

    # ==================================================================
    # ASPP MODULE
    # ==================================================================
    x = aspp_block(x, filters=256)

    # ==================================================================
    # DECODER
    # ==================================================================

    # 4× upsample to reach low-level feature resolution (output stride 4)
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear',
                            name='decoder_upsample1')(x)

    # Reduce low-level feature channels to 48  (as in the original paper)
    low_level_features = layers.Conv2D(
        48, 1, use_bias=False, name='decoder_low_level_conv'
    )(low_level_features)
    low_level_features = group_norm(name='decoder_low_level_gn')(low_level_features)
    low_level_features = layers.Activation('relu')(low_level_features)

    # Align spatial dims in case of any off-by-one from pooling
    def match_spatial_dims(tensors):
        high_level, low_level = tensors
        low_shape = tf.shape(low_level)
        return high_level[:, :low_shape[1], :low_shape[2], :], low_level

    x_matched, low_matched = layers.Lambda(
        match_spatial_dims, name='match_dims'
    )([x, low_level_features])

    # Fuse high-level and low-level features
    x = layers.Concatenate(name='decoder_concat')([x_matched, low_matched])

    x = layers.Conv2D(256, 3, padding='same', use_bias=False,
                      name='decoder_conv1')(x)
    x = group_norm(name='decoder_conv1_gn')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1, name='decoder_dropout1')(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False,
                      name='decoder_conv2')(x)
    x = group_norm(name='decoder_conv2_gn')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1, name='decoder_dropout2')(x)

    # Final 4× upsample back to original resolution
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear',
                            name='decoder_upsample2')(x)

    # ==================================================================
    # OUTPUT
    # ==================================================================
    if num_classes == 1:
        # Binary segmentation  →  sigmoid, single-channel mask
        outputs = layers.Conv2D(1, 1, activation='sigmoid', name='output')(x)
    else:
        # Multi-class segmentation  →  softmax over num_classes channels
        outputs = layers.Conv2D(num_classes, 1, activation='softmax',
                                name='output')(x)

    model = keras.Model(inputs, outputs, name='DeepLabV3Plus_ResNet50_GN')
    return model
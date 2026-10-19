from tensorflow import keras


@keras.utils.register_keras_serializable(package="roadseg")
class VGGPreprocess(keras.layers.Layer):
    """Preprocessing travels inside the saved model, including RGB→BGR."""

    def call(self, inputs):
        return keras.applications.vgg19.preprocess_input(inputs)


def build_model(config):
    layers = keras.layers
    inputs = keras.Input((config.height, config.width, 3), name="rgb_0_255")
    if config.backbone == "vgg19":
        processed = VGGPreprocess(name="vgg_preprocess")(inputs)
        backbone = keras.applications.VGG19(
            include_top=False,
            weights="imagenet" if config.imagenet_weights else None,
            input_tensor=processed,
        )
        backbone.trainable = config.train_backbone
        pool3 = backbone.get_layer("block3_pool").output
        pool4 = backbone.get_layer("block4_pool").output
        pool5 = backbone.get_layer("block5_pool").output
    else:
        x = layers.Rescaling(1 / 255, name="normalise_rgb")(inputs)
        pools = []
        for i, channels in enumerate((8, 16, 24, 32, 48), 1):
            x = layers.Conv2D(
                channels, 3, padding="same", activation="relu", name=f"block{i}_conv"
            )(x)
            x = layers.MaxPool2D(2, name=f"block{i}_pool")(x)
            pools.append(x)
        pool3, pool4, pool5 = pools[2:]
        if not config.train_backbone:
            # Decoder convolutions are added after freezing encoder layers.
            for layer in keras.Model(inputs, pool5).layers:
                layer.trainable = False
    x = layers.Conv2D(
        config.decoder_channels,
        3,
        padding="same",
        activation="relu",
        name="decoder_conv",
    )(pool5)
    x = layers.Conv2D(len(config.classes), 1, name="score32")(x)
    score16 = layers.Conv2D(len(config.classes), 1, name="score16")(pool4)
    x = layers.Resizing(
        config.height // 16, config.width // 16, interpolation="bilinear"
    )(x)
    x = layers.Add(name="fuse16")([x, score16])
    score8 = layers.Conv2D(len(config.classes), 1, name="score8")(pool3)
    x = layers.Resizing(
        config.height // 8, config.width // 8, interpolation="bilinear"
    )(x)
    x = layers.Add(name="fuse8")([x, score8])
    logits = layers.Resizing(
        config.height, config.width, interpolation="bilinear", name="logits"
    )(x)
    return keras.Model(inputs, logits, name=f"roadseg_fcn8_{config.backbone}")


def compile_model(model, config):
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="mean_with_sample_weight"
        ),
        weighted_metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy"),
            keras.metrics.MeanIoU(
                num_classes=len(config.classes),
                sparse_y_pred=False,
                name="mean_iou",
                dtype="float64",
            ),
        ],
    )
    return model

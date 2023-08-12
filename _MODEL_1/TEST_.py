from keras.models import load_model

model_path = "/Users/c/Desktop/AI/proto1/models/model_2023-08-12_11-44"
loaded_model = load_model(model_path)

print(loaded_model.summary())

# config = loaded_model.get_config()
# print(config)

# for layer in loaded_model.layers:
#     weights = layer.get_weights()
#     print(weights)


# optimizer_config = loaded_model.optimizer.get_config()
# print(optimizer_config)


# from keras.utils import plot_model
# plot_model(loaded_model, to_file='model.png', show_shapes=True)

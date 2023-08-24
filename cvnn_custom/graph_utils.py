import tensorflow as tf
from IPython.display import display_png


def get_dot_graph(model: tf.keras.Model, orientation="TD", file_save_name: str = None, **keras_kwargs):
    options = {"show_shapes": True, "show_layer_activations": True, "show_dtype": True}
    options.update(keras_kwargs)
    model_dot = tf.keras.utils.model_to_dot(model, **options)
    model_dot.set_ranksep(0.2)
    model_dot.set_pad(0.2)
    model_dot.set_orientation(orientation)

    nodelist = model_dot.get_node_list()
    for node in nodelist:
        node.set_fontsize(10)
        node.set_margin(0.03)
        node.set_height(0.03)
        node.set_fontname("Times")

    edgelist = model_dot.get_edge_list()
    for edge in edgelist:
        edge.set_arrowsize(0.5)

    if file_save_name is not None:
        if file_save_name[-4:] != ".png":
            file_save_name += ".png"
        model_dot.write(file_save_name, format="png")
        print(f"Graph of model saved at {file_save_name}")
    png = model_dot.create_png()
    display_png(png, raw=True)

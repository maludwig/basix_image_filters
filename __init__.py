from custom_nodes.basix_image_filters.hsl_nodes import HSL_NODES
from custom_nodes.basix_image_filters.hsv_nodes import HSV_NODES
from custom_nodes.basix_image_filters.rgb_nodes import RGB_NODES

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

ALL_NODES = HSL_NODES + RGB_NODES + HSV_NODES

for node_cls in ALL_NODES:
    uniq_node_name = "basix_image_filters__" + node_cls.__name__
    NODE_CLASS_MAPPINGS[uniq_node_name] = node_cls
    NODE_DISPLAY_NAME_MAPPINGS[uniq_node_name] = node_cls.UI_NAME

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

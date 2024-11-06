# comfyui_image_filters

This is a simple image filter library for ComfyUI. It requires no fancy requirements, simply using the basic
torch and numpy libraries. It is designed to be simple and easy to use. It is super helpful for hinting to 
the AI model that you want an image to be dark, with green light. Just feed it a darked, green image, with
0.7 - 0.9 denoise, and you'll get a darker, greener output.

All nodes take an image as input, and output an image.
The nodes also all take a mask as an optional input, if a mask is provided:
- And the masked_area is set to modify_unmasked, the image will be modified outside the mask
- And the masked_area is set to modify_masked, the image will be modified inside the mask

This way, you can easily take an image that is "ok except for the ___" and modify only the area you want to change.
This is built into ComfyUI (not part of this plugin), simply right-click on a Preview Image node, and Copy (Clipspace),
then make a Load Image node, and right-click that, and then Paste (Clipspace). You will now be able to right-click it,
and "Open in MaskEditor" to create a mask.
On Windows, Shift+Click-and-drag to zoom in and out, and Ctrl+click-and-drag to move the mask.
On OSX, try it out and let me know. I'm not rich enough to buy Apple hardware. haha.

### HSL Nodes

The HSL nodes are designed to be simple and easy to use.

Dragging the Workflow PNG into ComfyUI will automatically load the image as a workflow.
![HSL Workflow](docs/workflows/hsl_workflow.png)

#### Levels (RGB)

Similar to the HSL slider, this node allows you to adjust the levels of the image.
Adjusting a single color channel will adjust the lightness of the image only in that color,
for example, if you set a "midtones" value of 0.5 for the red channel, the image will be adjusted so that
it is more brightly red. This is very handy if you want a particular color to be more prominent in the image.

Options:
- "channel": The channel to adjust. "red", "green", "blue", or "all".


#### Levels (HSL)

Similar to the Photoshop Levels slider, this node allows you to adjust the levels of the image.

Options:
- "shadows": Any lightness value below this will be clipped to black. (0 - 1)
- "midtones": Applies a gamma correction to the midtones. Values above 2 will darken the image, values below 1 will lighten the image. (0 - 100)
- "highlights": Any lightness value above this will be clipped to white. (0 - 1)

#### Darken (HSL)

Darkens the image.

Options:
- "factor": The amount to darken the image. Shifts the lightness value towards black, a value of 0.5 will adjust the image so that pure white is 50% gray. A value of 1 will make the image completely black. Negative values will lighten the image.

#### Lighten (HSL)

Lightens the image.

Options:
- "factor": The amount to lighten the image. Shifts the lightness value towards white, a value of 0.5 will adjust the image so that pure black is 50% gray. A value of 1 will make the image completely white. Negative values will darken the image.

#### Saturate (HSL)

Saturates the image (adds color).

Options:
- "factor": The amount to saturate the image. A value of 0 will leave the image unchanged. A value of 1 will completely saturate the image. Negative values will desaturate the image.

#### Desaturate (HSL)

Desaturates the image (removes color).

Options:
- "factor": The amount to desaturate the image. A value of 0 will leave the image unchanged. A value of 1 will completely desaturate the image. Negative values will saturate the image.

#### Rotate Hue (HSL)

Rotates the hue of the image. Imagine a color wheel, with red at 0, green at 120, and blue at 240. This node will rotate the hue of the image.
This represents a color wheel of light. With red+blue being the complimentary color to green, this means that with a rotation of 180 degrees, red will become cyan, and blue will become yellow.
This is a very powerful tool for color correction, and can be used to create a wide variety of effects.
It is extremely useful if a model can do X, but it can only do X with a certain color. 
This node can be used to change the color of the image to match the color that the model is trained on.
For example, if a LoRA can do green shirts really well, but struggles to do red shirts, you can rotate the hue of the image from green to red, and the model will be able to do red shirts.
Best used with a mask, since otherwise it will also rotate the colors of the person wearing the shirt. This may not be necessary, since some models are able to do odd colors of skin,

Options:
- "degrees": The amount to rotate the hue of the image. A value of 0 will leave the image unchanged. A value of 180 will rotate the hue 180 degrees. Negative values will rotate the hue in the opposite direction.
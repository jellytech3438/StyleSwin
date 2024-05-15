import dearpygui.dearpygui as dpg
import numpy as np
from draggan import DragGAN
from array import array
import threading
import math

# np.set_printoptions(threshold=256)
add_point = 0
point_color = [(1, 0, 0), (0, 0, 1)]
mask_color = (0, 0, 0, 0.2)
points, steps = [], 0
dragging = False
# mvFormat_Float_rgb not currently supported on macOS
# More details: https://dearpygui.readthedocs.io/en/latest/documentation/textures.html#formats
texture_format = dpg.mvFormat_Float_rgba

# these are values that should change according to input weight size
mask = np.ones([256, 256], np.uint8)
image_size = 256
image_width, image_height, rgb_channel, rgba_channel = 256, 256, 3, 4
image_pixels = image_height * image_width
raw_data_size = image_width * image_height * rgba_channel
raw_data = array('f', [1] * raw_data_size)

model = DragGAN('cpu')

dpg.create_context()
dpg.create_viewport(title='DragGAN', width=1600, height=1600)

with dpg.texture_registry(show=False):
    dpg.add_raw_texture(
        width=image_width, height=image_height, default_value=raw_data,
        format=texture_format, tag="image"
    )


def update_image(new_image):
    # Convert image data (rgb) to raw_data (rgba)
    print("update on raw data: ", len(raw_data))
    for i in range(0, image_pixels):
        rd_base, im_base = i * rgba_channel, i * rgb_channel
        raw_data[rd_base:rd_base + rgb_channel] = array(
            'f', new_image[im_base:im_base + rgb_channel]
        )


def generate_image(sender, app_data, user_data):
    seed = dpg.get_value('seed')
    image = model.generate_image(seed)
    update_image(image)


def change_device(sender, app_data):
    model.to(app_data)


def set_lambda_mask(sender, app_data):
    mask = dpg.get_value("lambda_mask")
    model.lambda_mask = mask


def dragging_thread():
    global points, steps, dragging, mask
    while (dragging):
        status, ret, sfeatures, simage = model.step(points, mask)
        if status:
            points, image = ret
        else:
            dragging = False
            return
        update_image(image)
        for i in range(len(points)):
            draw_point(*points[i], point_color[i % 2])
        steps += 1
        
        # store features map into features folder every 5 steps
        if steps % 5 == 0:
            model.store_feature(steps, simage, sfeatures)

        dpg.set_value('steps', f'steps: {steps}')


width, height = 260, 200
posx, posy = 0, 0
with dpg.window(
    label='Network & Latent', width=width, height=height, pos=(posx, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    dpg.add_text('device', pos=(5, 20))
    dpg.add_combo(
        ('cpu', 'cuda'), default_value='cpu', width=60, pos=(70, 20),
        callback=change_device,
    )

    dpg.add_text('weight', pos=(5, 40))

    def select_cb(sender, app_data):
        selections = app_data['selections']
        if selections:
            for fn in selections:
                model.load_ckpt(selections[fn])
                # store image size to a variable to avoid if-else logic
                global image_size, mask, image_width, image_height, image_pixels, raw_data_size, raw_data
                image_size = model.input_image_size
                mask = np.ones([image_size, image_size], np.uint8)
                image_width, image_height = image_size, image_size
                image_pixels = image_height * image_width
                raw_data_size = image_width * image_height * rgba_channel
                raw_data = array('f', [1] * raw_data_size)


                dpg.configure_item("image", width=image_width, height=image_height, default_value=raw_data,
                    format=texture_format, tag="image"
                )


                # dpg.set_item_width("Image Win", dpg.get_item_width("image"))
                # dpg.set_item_height("Image Win", dpg.get_item_height("image"))
                # dpg.configure_item("Image Win", window=int(dpg.get_item_width("image")), height=int(dpg.get_item_height("image")), label='Image')
                # dpg.delete_item("Image Win")
                # with dpg.window(
                #     label='Image', pos=(posx, posy), tag='Image Win',
                #     no_move=True, no_close=True, no_collapse=True, no_resize=True,
                # ):
                #     dpg.add_image("image", show=True, tag='image_data', pos=(10, 30), width=int(
                #         dpg.get_item_width("image") * scale), height=int(dpg.get_item_height("image") * scale))
                    

                break

    def cancel_cb(sender, app_data):
        ...

    with dpg.file_dialog(
        directory_selector=False, show=False, callback=select_cb, id='weight selector',
        cancel_callback=cancel_cb, width=700, height=400
    ):
        dpg.add_file_extension('.*')
    dpg.add_button(
        label="select weight", callback=lambda: dpg.show_item("weight selector"),
        pos=(70, 40),
    )

    dpg.add_text('latent', pos=(5, 60))
    dpg.add_input_int(
        label='seed', width=100, pos=(70, 60), tag='seed', default_value=512,
    )
    dpg.add_input_float(
        label='step size', width=54, pos=(70, 80), step=-1, default_value=0.002,
    )
    dpg.add_button(label="reset", width=54, pos=(70, 100), callback=None)
    dpg.add_radio_button(
        items=('w', 'w+'), pos=(130, 100), horizontal=True, default_value='w+',
    )
    dpg.add_button(label="generate", pos=(70, 120), callback=generate_image)

posy += height + 2
with dpg.window(
    label='Drag', width=width, height=height, pos=(posx, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    def add_point_cb():
        global add_point
        add_point += 2

    def reset_point_cb():
        global points
        points = []

    def start_cb():
        global dragging
        if dragging:
            return
        dragging = True
        threading.Thread(target=dragging_thread).start()

    def stop_cb():
        global dragging
        dragging = False
        print('stop dragging...')

    def reset_mask():
        global mask
        mask = np.fill(1)

    def print_mask():
        global mask
        np.set_printoptions(threshold=300000000000, linewidth=256*3)
        print(str(mask).replace(' ', ''))

    dpg.add_text('drag', pos=(5, 20))
    dpg.add_button(label="add point", width=80,
                   pos=(70, 20), callback=add_point_cb)
    dpg.add_button(label="reset point", width=80,
                   pos=(155, 20), callback=reset_point_cb)
    dpg.add_button(label="start", width=80, pos=(70, 40), callback=start_cb)
    dpg.add_button(label="stop", width=80, pos=(155, 40), callback=stop_cb)
    dpg.add_text('steps: 0', tag='steps', pos=(70, 60))

    dpg.add_text('mask', pos=(5, 80))
    dpg.add_button(label="fixed area", width=80, pos=(70, 80), callback=None)
    dpg.add_button(label="reset mask", width=80,
                   pos=(70, 100), callback=reset_mask)
    # debug used
    dpg.add_button(label="print mask", width=80,
                   pos=(70, 120), callback=print_mask)
    dpg.add_checkbox(label='show mask', tag='show_mask',
                     pos=(155, 100), default_value=False)
    dpg.add_input_int(label='radius', width=100, tag="radius",
                      pos=(70, 140), default_value=20)
    dpg.add_input_float(label='lambda', width=100, tag="lambda_mask",
                        pos=(70, 160), default_value=20, callback=set_lambda_mask)

posy += height + 2
with dpg.window(
    label='Capture', width=width, height=height, pos=(posx, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    dpg.add_text('capture', pos=(5, 20))
    dpg.add_input_text(pos=(70, 20), default_value='capture')
    dpg.add_button(label="save image", width=80, pos=(70, 40), callback=None)


def draw_point(x, y, color):
    global raw_data, image_width, image_height
    x_start, x_end = max(0, x - 2), min(image_width, x + 2)
    y_start, y_end = max(0, y - 2), min(image_height, y + 2)
    print(x, y, x_start, x_end, y_start, y_end)
    for x in range(x_start, x_end):
        for y in range(y_start, y_end):
            offset = (y * image_width + x) * rgba_channel
            raw_data[offset:offset +
                     rgb_channel] = array('f', color[:rgb_channel])


def append_to_mask(x, y):
    global mask, raw_data, image_width, image_height
    r = dpg.get_value("radius")
    x_start, x_end = max(0, x - r), min(image_width, x + r)
    y_start, y_end = max(0, y - r), min(image_height, y + r)
    for xi in range(x_start, x_end):
        for yi in range(y_start, y_end):
            if math.sqrt(abs(xi-x)**2 + abs(yi-y)**2) > r:
                continue
            # change to the value that maks the tensor
            mask[yi, xi] = 0
            # rawdata should add transparent color
            offset = (yi * image_width + xi) * rgba_channel
            raw_data[offset:offset + rgba_channel] = array('f', mask_color)


def select_point(sender, app_data):
    global add_point, points
    if add_point <= 0:
        return
    ms_pos = dpg.get_mouse_pos(local=False)
    id_pos = dpg.get_item_pos('image_data')
    iw_pos = dpg.get_item_pos('Image Win')
    ix = int(ms_pos[0]-id_pos[0]-iw_pos[0])
    iy = int(ms_pos[1]-id_pos[1]-iw_pos[1])
    draw_point(ix, iy, point_color[add_point % 2])
    points.append(np.array([ix, iy]))
    print(points)
    add_point -= 1


def draw_mask(sender, app_data):
    global mask, mask_color
    ms_pos = dpg.get_mouse_pos(local=False)
    id_pos = dpg.get_item_pos('image_data')
    iw_pos = dpg.get_item_pos('Image Win')
    ix = int(ms_pos[0]-id_pos[0]-iw_pos[0])
    iy = int(ms_pos[1]-id_pos[1]-iw_pos[1])
    if ix < 0:
        ix = 0
    elif ix > image_width-1:
        ix = image_width-1
    if iy < 0:
        iy = 0
    elif iy > image_height-1:
        iy = image_height-1
    append_to_mask(ix, iy)


posx, posy = 2 + width, 0
scale = 1.0
with dpg.window(
    label='Image', pos=(posx, posy), tag='Image Win',
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    dpg.add_image("image", show=True, tag='image_data', pos=(10, 30), width=int(
        dpg.get_item_width("image") * scale), height=int(dpg.get_item_height("image") * scale))

with dpg.item_handler_registry(tag='double_clicked_handler'):
    dpg.add_item_double_clicked_handler(callback=select_point)

with dpg.handler_registry(tag='drag_handler'):
    dpg.add_mouse_drag_handler(
        button=dpg.mvMouseButton_Right, callback=draw_mask)

dpg.bind_item_handler_registry("image_data", "double_clicked_handler")
# dpg.bind_item_handler_registry("imgae_data", "drag_handler")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
# while dpg.is_dearpygui_running():
#     dpg.set_item_width("Image Win", image_size)
#     dpg.set_item_height("Image Win", image_size)
#
#     dpg.render_dearpygui_frame()

dpg.destroy_context()

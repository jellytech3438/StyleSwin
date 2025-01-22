import dearpygui.dearpygui as dpg
import numpy as np
from draggan import DragGAN
# from freedrag import DragGAN
from array import array
import threading
import math
from PIL import Image

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
origin_image = array('f', [1] * raw_data_size)
store_image = array('f', [1] * raw_data_size)
device = "cuda"
store_feat_step = 5

model = DragGAN(device)

dpg.create_context()
dpg.create_viewport(title='mm-final', width=1600, height=1600)

with dpg.texture_registry(show=False):
    dpg.add_raw_texture(
        width=image_width, height=image_height, default_value=raw_data,
        format=texture_format, tag="image"
    )


def update_image(new_image):
    # Convert image data (rgb) to raw_data (rgba)
    for i in range(0, image_pixels):
        rd_base, im_base = i * rgba_channel, i * rgb_channel
        raw_data[rd_base:rd_base + rgb_channel] = array(
            'f', new_image[im_base:im_base + rgb_channel]
        )


def update_save_image(new_image):
    for i in range(0, image_pixels):
        rd_base, im_base = i * rgba_channel, i * rgb_channel
        store_image[rd_base:rd_base + rgb_channel] = array(
            'f', new_image[im_base:im_base + rgb_channel]
        )


def generate_image(sender, app_data, user_data):
    global raw_data, origin_image
    seed = dpg.get_value('seed')
    image = model.generate_image(seed)
    update_image(image)
    # update image will change origin_image, so only in generate_image set the origin
    for i in range(0, image_pixels):
        rd_base, im_base = i * rgba_channel, i * rgb_channel
        origin_image[rd_base:rd_base + rgb_channel] = array(
            'f', image[im_base:im_base + rgb_channel]
        )


def change_device(sender, app_data):
    global device
    model.to(app_data)
    device = app_data


def set_lambda_mask(sender, app_data):
    mask = dpg.get_value("lambda_mask")
    model.lambda_mask = mask


def dragging_thread():
    global points, steps, dragging, mask, plot_loss, store_image
    while (dragging):
        status, ret, sfeatures, simage = model.step(
            points, mask, steps, visiualize_attention=dpg.get_value("vis attention"))
        if status:
            points, image = ret
        else:
            dragging = False
            print("reach target points")
            break
        update_image(image)
        update_save_image(image)
        for i in range(len(points)):
            draw_point(*points[i], point_color[i % 2])
        steps += 1

        # store features map into features folder every n steps
        if dpg.get_value("store feature") and steps % store_feat_step == 0:
            model.store_feature(steps, simage, sfeatures)
        dpg.set_value('steps', f'steps: {steps}')

    # plot loss
    if dpg.get_value("plot loss"):
        model.plot_loss()
        print("saves loss ploting")

    # call calculate score function after dragging
    lpips = model.lpip_score(origin_image, raw_data)
    mds = model.mean_distance_score(origin_image, raw_data, points)
    dpg.set_value('lpips', f'lpips: {lpips:.5f}')
    dpg.set_value('mds', f'mds: {mds:.5f}')
    print("lpips:", lpips)
    print("mds:", mds)


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


width, height = 256, 256
posx, posy = 0, 0
with dpg.window(
    label='Network & Latent', width=width, height=height, pos=(posx, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    dpg.add_text('device', pos=(5, 20))
    dpg.add_combo(
        ('cpu', 'cuda'), default_value=device, width=60, pos=(70, 20),
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
                                   format=texture_format, tag="image")

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
    dpg.add_text('image', pos=(5, 140))
    dpg.add_button(label="select image", pos=(70, 140), callback=None)
    dpg.add_checkbox(label='store feature', tag='store feature',
                     pos=(70, 160), default_value=False)
    dpg.add_checkbox(label='plot loss', tag='plot loss',
                     pos=(70, 180), default_value=False)
    dpg.add_checkbox(label='vis attention', tag='vis attention',
                     pos=(70, 200), default_value=False)

posy += height + 2
with dpg.window(
    label='Drag & Score', width=width, height=height, pos=(posx, posy),
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
        for y in range(256):
            for x in range(256):
                if mask[y, x] == 0:
                    mask[y, x] = 1
                    raw_data[256*y*rgba_channel + x*rgba_channel] = 0.
        # print(np.count_nonzero(mask))

    def print_mask():
        global mask
        np.set_printoptions(threshold=300000000000, linewidth=256*3)
        # print(str(mask).replace(' ', ''))

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
    dpg.add_text('score', pos=(5, 180))
    dpg.add_text('lpips: 0', tag='lpips', pos=(70, 180))
    dpg.add_text('mds: 0', tag='mds', pos=(70, 200))

posy += height + 2


def save_image(sender, app_data):
    global store_image, raw_data, origin_image, image_width, image_height
    temp = np.array(origin_image) * 255
    # temp = np.array(store_image) * 255
    # temp = np.array(raw_data) * 255
    im = Image.fromarray(temp.astype(
        np.uint8).reshape((height, width, 4)))
    im.save("imgs/"+dpg.get_value("save image name")+'.png',)
    print("save image successfully")


def save_points(sender, app_data):
    global points
    if len(points) == 0:
        print("no points selected")
        return 0
    np.save("points/latest_points", np.array(points))
    print("save points successfully")


def save_mask(sender, app_data):
    global mask
    np.save("mask/latest_mask", np.array(mask))
    print("save mask successfully")


with dpg.window(
    label='Capture & Points', width=width, height=height, pos=(posx, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    dpg.add_text('capture', pos=(5, 20))
    dpg.add_input_text(pos=(70, 20), default_value='capture',
                       tag='save image name')
    dpg.add_button(label="save image", width=80,
                   pos=(70, 40), callback=save_image)

    dpg.add_text('points', pos=(5, 60))
    dpg.add_button(
        label="save points", width=100, pos=(70, 60), callback=save_points
    )

    def select_pt(sender, app_data):
        selections = app_data['selections']
        if selections:
            for fn in selections:
                global points, add_point
                points = np.load(selections[fn]).tolist()
                for i in range(len(points)):
                    add_point += 1
                    if i % 2 == 0:
                        draw_point(points[i][0], points[i][1], point_color[0])
                    else:
                        draw_point(points[i][0], points[i][1], point_color[1])

                print("load points successfully")
                break

    def cancel_pt(sender, app_data):
        ...

    with dpg.file_dialog(
        directory_selector=False, show=False, callback=select_pt, id='points selector',
        cancel_callback=cancel_pt, width=700, height=400
    ):
        dpg.add_file_extension('.*')
    dpg.add_button(
        label="load points", width=100, callback=lambda: dpg.show_item("points selector"),
        pos=(70, 80),
    )

    dpg.add_text('mask', pos=(5, 100))
    dpg.add_button(
        label="save mask", width=100, pos=(70, 100), callback=save_mask
    )

    def select_mask(sender, app_data):
        selections = app_data['selections']
        if selections:
            for fn in selections:
                global mask
                select_mask = np.load(selections[fn]).tolist()
                mask = np.array(select_mask)
                print(mask.shape)
                print("load mask successfully")
                break

    def cancel_mask(sender, app_data):
        ...

    with dpg.file_dialog(
        directory_selector=False, show=False, callback=select_mask, id='mask selector',
        cancel_callback=cancel_pt, width=700, height=400
    ):
        dpg.add_file_extension('.*')
    dpg.add_button(
        label="load mask", width=100, callback=lambda: dpg.show_item("mask selector"),
        pos=(70, 120),
    )


def draw_point(x, y, color):
    global raw_data, image_width, image_height
    x_start, x_end = max(0, x - 2), min(image_width, x + 2)
    y_start, y_end = max(0, y - 2), min(image_height, y + 2)
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
    no_move=True, no_close=True, no_collapse=True, no_resize=False,
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


og_loss = []
mask_loss = []


def update_series():
    og_loss_x = []
    og_loss_y = []
    mask_loss_y = []
    mask_loss_x = []
    for i in range(0, 500):
        og_loss_y.append(i / 1000)
        mask_loss_y.append(0.5 + 0.5 * math.cos(50 * i / 1000))
    dpg.set_value('series_tag', [og_loss_y, mask_loss_y])
    dpg.set_item_label('series_tag', "loss series through step")


# with dpg.window(label="Loss plot", pos=(posx+256, posy), tag="loss"):
#     # create plot
#     with dpg.plot(label="Line Series", height=800, width=800):
#         dpg.add_button(label="Update Series", callback=update_series)
#
#         # optionally create legend
#         dpg.add_plot_legend()
#
#         # REQUIRED: create x and y axes
#         dpg.add_plot_axis(dpg.mvXAxis, label="step")
#         dpg.add_plot_axis(dpg.mvYAxis, label="loss", tag="loss_axis")
#
#         # series belong to a y axis
#         dpg.add_line_series(og_loss, mask_loss,
#                             label="loss series through step", parent="loss_axis")


dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

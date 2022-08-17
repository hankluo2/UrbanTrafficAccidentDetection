from datamaker.carla_vehicle_annotator import *


def save_custom_output(
        world,  # carla world
        carla_img,
        vehicle_list,  # filtered
        bboxes,  #
        old_vehicle_list=None,  # removed
        old_bboxes=None,  #
        cc_rgb=carla.ColorConverter.Raw,
        path='',
        save_patched=False,
        add_data=None,
        out_format='json'):
    """Redesign output saver.

    """

    # save rgb frame
    carla_img.save_to_disk(path + 'out_rgb/%06d.png' % carla_img.frame, cc_rgb)

    out_dict = dict()

    bboxes_list = [bbox.tolist() for bbox in bboxes]
    out_dict['bboxes'] = bboxes_list
    out_dict['vehicles'] = vehicle_list

    frame = world.get_snapshot().frame
    timestamp = world.get_snapshot().timestamp
    timestamp = timestamp.elapsed_seconds

    frame_dict = dict()
    timestamp_dict = dict()
    location = dict()
    velocity = dict()
    angular_velocity = dict()
    accelaration = dict()
    bbox_dict = dict()
    vid_dict = dict()
    vid_list = []

    for v, b in zip(vehicle_list, bboxes_list):
        v = world.get_actor(v.id)
        vid_list.append(v.id)
        vid_dict[v.id] = v.id

        frame_dict[v.id] = frame
        timestamp_dict[v.id] = timestamp

        bbox_dict[v.id] = b
        location[v.id] = {'x': v.get_location().x, 'y': v.get_location().y, 'z': v.get_location().z}
        velocity[v.id] = {'x': v.get_velocity().x, 'y': v.get_velocity().y, 'z': v.get_velocity().z}
        angular_velocity[v.id] = {
            'x': v.get_angular_velocity().x,
            'y': v.get_angular_velocity().y,
            'z': v.get_angular_velocity().z
        }
        accelaration[v.id] = {'x': v.get_acceleration().x, 'y': v.get_acceleration().y, 'z': v.get_acceleration().z}

    frame_info = {
        'vehicle_ids': vid_list,
        'vehicle_id_dict': vid_dict,
        'frame_ids': frame_dict,
        'timestamp': timestamp_dict,
        'bboxes': bbox_dict,
        'locations': location,
        'velocities': velocity,
        'angular_velocities': angular_velocity,
        'accelaration': accelaration
    }

    # if old_bboxes is not None:
    #     old_bboxes_list = [bbox.tolist() for bbox in old_bboxes]
    #     out_dict['removed_bboxes'] = old_bboxes_list
    # if add_data is not None:
    #     out_dict['others'] = add_data

    if out_format == 'json':
        filename = path + 'out_bbox/%06d.json' % carla_img.frame
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, 'w') as outfile:
            json.dump(frame_info, outfile, indent=4)

    if save_patched:
        carla_img.convert(cc_rgb)
        img_bgra = np.array(carla_img.raw_data).reshape((carla_img.height, carla_img.width, 4))
        img_rgb = np.zeros((carla_img.height, carla_img.width, 3))

        img_rgb[:, :, 0] = img_bgra[:, :, 2]
        img_rgb[:, :, 1] = img_bgra[:, :, 1]
        img_rgb[:, :, 2] = img_bgra[:, :, 0]
        img_rgb = np.uint8(img_rgb)
        image = Image.fromarray(img_rgb, 'RGB')
        img_draw = ImageDraw.Draw(image)
        for crop in bboxes:
            u1 = int(crop[0, 0])
            v1 = int(crop[0, 1])
            u2 = int(crop[1, 0])
            v2 = int(crop[1, 1])
            crop_bbox = [(u1, v1), (u2, v2)]
            img_draw.rectangle(crop_bbox, outline="red")
        filename = path + 'out_rgb_bbox/%06d.jpg' % carla_img.frame
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        image.save(filename)

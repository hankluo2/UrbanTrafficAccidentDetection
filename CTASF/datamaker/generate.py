import carla
import os
import gc
from carla import VehicleLightState as vls
from agents import BasicAgent

import datamaker.carla_vehicle_annotator as cva
import datamaker.cva_custom as cva_custom
from datamaker.utils import exp_dir, retrieve_data
from datamaker.settings import Weather, Location

import logging
import random
import time
import queue
import numpy as np
from pathlib import Path


class CustomAgent(BasicAgent):

    def __init__(self, vehicle, target_speed=72, it=0, debug=False):
        """
        :param target_speed: speed (in Km/h) at which the vehicle will move
        """
        super().__init__(vehicle, target_speed=target_speed, opt_dict={})
        self.it = it

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """
        # Actions to take during each simulation step
        control = carla.VehicleControl()

        # custom vehicle control at intersections
        control.steer = 0.0

        control.throttle = 1.0 - self.it * 0.04 if self.it < 25 else 0.0
        control.brake = self.it * 0.04 if self.it < 25 else 1.0

        control.hand_brake = False
        control.manual_gear_shift = False

        return control


def launch(cfg):
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.INFO)

    if cfg.weather_on:
        save_dir = str(
            Path(cfg.dataset) / ('scene' + cfg.scene_num + cfg.weather))
    else:
        save_dir = str(Path(cfg.dataset) / ('scene' + cfg.scene_num))

    cur_dir = exp_dir(save_dir=save_dir) + '/'
    print(cur_dir)

    vehicles_list = []
    nonvehicles_list = []

    client = carla.Client(cfg.host, cfg.port)
    client.set_timeout(10.0)
    random.seed(time.time())
    synchronous_master = False

    try:
        traffic_manager = client.get_trafficmanager(cfg.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        world = client.get_world()

        # set weather
        if cfg.weather_on:
            weather = Weather(world.get_weather(), cfg)
            world.set_weather(weather.weather)

        print('RUNNING in synchronous mode:')
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        tick_gap = 1. / cfg.fps
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = tick_gap
            world.apply_settings(settings)
        else:
            synchronous_master = False

        blueprints = world.get_blueprint_library().filter('vehicle.*')
        # ============================ filter special vehicles ==============================
        blueprints = [
            x for x in blueprints
            if int(x.get_attribute('number_of_wheels')) == 4
        ]
        blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
        blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)
        # ===================================================================================

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if cfg.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif cfg.number_of_vehicles > number_of_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, cfg.number_of_vehicles,
                            number_of_spawn_points)
            cfg.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= cfg.number_of_vehicles:
                break
            blueprint = random.choice(
                blueprints)  # randomly select a vehicle blueprint
            if blueprint.has_attribute('color'):
                color = random.choice(
                    blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(
                    blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            light_state = vls.NONE
            if cfg.car_lights_on == 1:
                light_state = vls.Position | vls.LowBeam | vls.LowBeam

            # spawn the cars and set their autopilot and light state all together
            batch.append(
                SpawnActor(blueprint, transform).then(
                    SetAutopilot(FutureActor, True,
                                 traffic_manager.get_port())).then(
                                     SetVehicleLightState(
                                         FutureActor, light_state)))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        print('Created %d npc vehicles.' % len(vehicles_list))

        # ============================== set hero vehicle here!!! =====================
        danger_vid = random.sample(vehicles_list,
                                   int(len(vehicles_list) * cfg.hero_pct))
        for v in danger_vid:
            danger_vehicle = world.get_actor(v)
            traffic_manager.auto_lane_change(danger_vehicle, False)
            traffic_manager.ignore_lights_percentage(danger_vehicle, 100)
            traffic_manager.distance_to_leading_vehicle(danger_vehicle, 0)
            traffic_manager.vehicle_percentage_speed_difference(
                danger_vehicle, -200)
            traffic_manager.ignore_signs_percentage(danger_vehicle, 100)
            traffic_manager.ignore_vehicles_percentage(danger_vehicle, 100)
            traffic_manager.ignore_walkers_percentage(danger_vehicle, 100)
            traffic_manager.set_percentage_keep_right_rule(danger_vehicle, 100)

        # ============================= hero vehicle set ==============================

        # Spawn sensors
        q_list = []
        idx = 0
        tick_queue = queue.Queue()
        world.on_tick(tick_queue.put)
        q_list.append(tick_queue)
        tick_idx = idx
        idx += 1

        # Spawn RGB camera
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('sensor_tick', str(tick_gap))
        cam_bp.set_attribute('image_size_x', str(cfg.width))
        cam_bp.set_attribute('image_size_y', str(cfg.height))
        cam_bp.set_attribute('fov', str(cfg.fov))
        cam_location = Location(cfg.scene_num)
        cam_transform = carla.Transform(
            carla.Location(x=cam_location.x,
                           y=cam_location.y,
                           z=cam_location.z),
            carla.Rotation(pitch=cam_location.pitch, yaw=cam_location.yaw))
        cam = world.spawn_actor(cam_bp, cam_transform)

        nonvehicles_list.append(cam)
        cam_queue = queue.Queue()
        cam.listen(cam_queue.put)
        q_list.append(cam_queue)
        cam_idx = idx
        idx = idx + 1
        print('RGB camera ready!')

        # Spawn depth camera
        depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute('sensor_tick', str(tick_gap))
        depth = world.spawn_actor(depth_bp, cam_transform)

        # cc_depth_log = carla.ColorConverter.LogarithmicDepth
        nonvehicles_list.append(depth)
        depth_queue = queue.Queue()
        depth.listen(depth_queue.put)
        q_list.append(depth_queue)
        depth_idx = idx
        idx = idx + 1
        print('Depth camera ready!')

        # Begin the loop
        time_sim = 0
        # fps = 1 / cfg.tick_gap
        fps = cfg.fps

        hero_in_roi = {}
        for _ in range(int(60 * cfg.simul_min * fps)):
            # Extract the available data
            now_frame = world.tick()

            if time_sim >= tick_gap:
                data = [retrieve_data(q, now_frame) for q in q_list]
                assert all(x.frame == now_frame for x in data if x is not None)

                # Skip if any sensor data is not available
                if None in data:
                    # print("No sensor data available. Continue.")
                    continue

                vehicles_raw = world.get_actors().filter('vehicle.*')
                snap = data[tick_idx]
                rgb_img = data[cam_idx]
                depth_img = data[depth_idx]

                # Attach additional information to the snapshot
                vehicles = cva.snap_processing(vehicles_raw, snap)

                # Save depth image, RGB image, and Bounding Boxes data
                depth_meter = cva.extract_depth(depth_img)
                filtered, removed = cva.auto_annotate(vehicles, cam,
                                                      depth_meter)

                filtered_vehicles = filtered['vehicles']
                filtered_vehicles_ids = [fv.id for fv in filtered_vehicles]

                # Judge whether danger vehicle in ROI:
                danger_roi_ids = inter(filtered_vehicles_ids, danger_vid)

                if danger_roi_ids:
                    for hero_id in danger_roi_ids:
                        # get each danger car id in roi: hero_id
                        hero = world.get_actor(hero_id)
                        if hero_id not in hero_in_roi:
                            hero_in_roi[hero_id] = {}
                            hero_in_roi[hero_id]['iter'] = 0
                        else:
                            hero_in_roi[hero_id]['iter'] += 1

                        hero_agent = CustomAgent(
                            hero, it=hero_in_roi[hero_id]['iter'])
                        hero_in_roi[hero_id]['agent'] = hero_agent

                    for hero_id in hero_in_roi:
                        agent = hero_in_roi[hero_id]['agent']
                        hero.apply_control(agent.run_step())

                        del agent

                # Examples for vehicle status info reporting:
                # for v in filtered_vehicles:
                #     v = world.get_actor(v.id)
                #     print(v.get_location())
                #     print(v.get_velocity().y)
                #     print(v.get_angular_velocity().y)
                #     print(v.get_acceleration().y)

                cva_custom.save_custom_output(world,
                                              rgb_img,
                                              filtered_vehicles,
                                              filtered['bbox'],
                                              save_patched=False,
                                              path=cur_dir,
                                              out_format='json')
                time_sim

            time_sim += settings.fixed_delta_seconds

    except:
        print("\nFailed to enter the main function. Exit.\n")
    try:
        cam.stop()
        depth.stop()
    except:
        print("Simulation ended before sensors have been created")

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    print('destroying %d nonvehicles' % len(nonvehicles_list))
    client.apply_batch(
        [carla.command.DestroyActor(x) for x in nonvehicles_list])

    print('\ndestroying %d vehicles' % len(vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    gc.collect()
    time.sleep(0.5)


def make_data(cfg):
    try:
        for i in range(cfg.video_num):
            since = time.time()

            try:
                launch(cfg)
            except KeyboardInterrupt:
                os.system(f"docker stop {cfg.dockerid}")
                print(f"Carla simulator docker#{cfg.dockerid} terminated.")
                return
            finally:
                print('\ndone.')
                print("Simulation cost {:.4f} seconds.\n".format(time.time() -
                                                                 since))
    except:
        print("Program terminated. Exit.")


def inter(a, b):
    return list(set(a) & set(b))

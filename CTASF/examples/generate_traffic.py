#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""Example script to generate traffic in the simulation"""

import glob
import os
import sys
import time
from pathlib import Path
from queue import Queue, Empty
import argparse
import logging

from numpy import random
import pandas as pd

import carla
from carla import VehicleLightState as vls

from point_in_polygon import PolygonOperator as PO


def unfold_array(data, data_name):
    """unfold data if data is not a array-like type.

    Args:
        data (basic type or class): unkown
    """
    basic = [int, float, str]
    for dataType in basic:
        if isinstance(data, dataType):
            return {data_name: data}
    # else
    var_list = [var for var in dir(data) if '_' not in var]  # add conditions here
    data_preffix = '_' + type(data).__name__ + '_'  # str
    unfold_data_dict = dict()
    for var in var_list:
        x = getattr(data, var)  # get data.var
        unfold_data_dict[data_name + data_preffix + var] = x
    return unfold_data_dict


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def sensor_callback(sensor_data, sensor_queue, sensor_name, location_dict):
    """Callback function to listen data from simulator.
        Do stuff with the sensor_data data like save it to disk
        Then you just need to add to the queue

    Args:
        sensor_data (data): data listened
        sensor_queue (queue.Queue()): pythonAPI outer vessel
        sensor_name (str): id belongs to actor
    """
    # print(sensor_data.frame, sensor_name, str(sensor_data))
    # sensor_queue.put((sensor_data.frame, sensor_name, str(sensor_data)))
    sensor_queue.put((sensor_data.frame, sensor_name, type(sensor_data).__name__, sensor_data, location_dict))


def exp_dir(prfx: str, save_dir='D:\\ExperimentData') -> str:
    """Get output directory string in current workspace.

    Args:
        prfx (str): preffix of the saving derectory.
        save_dir (str): directory to save experiment data.

    Returns:
        str: directory to save output data.
    """
    save_path = Path(save_dir)
    dir_list = [str(dir_) for dir_ in save_path.glob(prfx + '*')]
    dir_list.sort(key=lambda s: int(s.split('-')[-1]))
    next_exp_num = int(dir_list[-1].split('-')[-1]) + 1
    new_dir = dir_list[-1].split('-')[-2] + '-' + str(next_exp_num)

    return new_dir


def location_in_region_of_interest(x: float, y: float, regions: list) -> bool:
    """Return whether location is in surveillance region or not.

    Args:
        x (float): x position in map
        y (float): y position in map
        regions (list): of PolygonOperators, determine whether x, y is in polygon

    Returns:
        bool: True or False
    """
    for region in regions:
        if region.is_in_polygon(x, y):
            return True
    return False


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-n',
                           '--number-of-vehicles',
                           metavar='N',
                           default=30,
                           type=int,
                           help='Number of vehicles (default: 30)')
    argparser.add_argument('-w',
                           '--number-of-walkers',
                           metavar='W',
                           default=10,
                           type=int,
                           help='Number of walkers (default: 10)')
    argparser.add_argument('--safe', action='store_true', help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument('--filterv',
                           metavar='PATTERN',
                           default='vehicle.*',
                           help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument('--generationv',
                           metavar='G',
                           default='All',
                           help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument('--filterw',
                           metavar='PATTERN',
                           default='walker.pedestrian.*',
                           help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument('--generationw',
                           metavar='G',
                           default='2',
                           help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument('--tm-port', metavar='P', default=8000, type=int, help='Port to communicate with TM (default: 8000)')
    argparser.add_argument('--asynch', action='store_true', help='Activate asynchronous mode execution')
    argparser.add_argument('--hybrid', action='store_true', help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument('-s',
                           '--seed',
                           metavar='S',
                           type=int,
                           help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument('--car-lights-on', action='store_true', default=False, help='Enable car lights')
    argparser.add_argument('--hero', action='store_true', default=False, help='Set one of the vehicles as hero')
    argparser.add_argument('--respawn',
                           action='store_true',
                           default=False,
                           help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument('--no-rendering', action='store_true', default=False, help='Activate no rendering mode')

    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    preffix = '_out'  # preffix of experiment examples
    output_dir = exp_dir(preffix)
    print("Experiment data saved at {}".format(output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        world = client.get_world()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        # original_settings = world.get_settings()
        settings = world.get_settings()

        if not args.asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        tick_gap = 0.05  # set ticking gap
        settings.fixed_delta_seconds = tick_gap
        if args.no_rendering:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        sensor_queue = Queue()
        # Blueprints setup
        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        blueprint_library = world.get_blueprint_library()

        # TODO (2021/9/28): set various kinds of sensor [blueprints] hereunder
        imu_bp = blueprint_library.find('sensor.other.imu')
        radar_bp = blueprint_library.find('sensor.other.radar')
        collision_bp = blueprint_library.find('sensor.other.collision')
        # lidar
        # RSS

        # ------------------
        # Set surveillance camera
        # -------------------
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '75')
        camera_bp.set_attribute('sensor_tick', str(tick_gap))  # mind the ticking gap
        camera_transform = carla.Transform(carla.Location(x=-66, y=145, z=7), carla.Rotation(yaw=-45))
        camera = world.spawn_actor(camera_bp, camera_transform)
        print('created %s' % camera.type_id)
        # set region of interest
        regionX1, regionX2 = [-37.3, -37.3, -55.4, -55.4], [25.1, 22.7, -55.4, -63.9]
        regionY1, regionY2 = [9.0, 124.4, 124.4, 69.2], [124.4, 140.8, 140.8, 129.3]
        region1 = PO(regionY1, regionX1)  # mind that in carla, x, y is not Cartesian, which should be converted
        region2 = PO(regionY2, regionX2)
        regions = [region1, region2]

        cc = carla.ColorConverter.Raw
        camera.listen(lambda image: image.save_to_disk(output_dir + '/{}.jpg'.format(image.frame), cc))
        # --------------------- monitor set ----------------------------

        blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # TODO: (2021/9/30): spawn abnormal vehicles manually
        #   Set several vehicles in ROI, then tick 100s. And then check output
        #   images.
        # abnormal = True
        # abnormals = []
        # # ---------- set abnormal ---------------
        # if abnormal:
        #     print("setting abn_veh:")
        #     for _ in range(10):
        #         for n, transform in enumerate(spawn_points):
        #             # print(transform)
        #             print("first:", n)
        #             y, x, z = transform.location.x, transform.location.y, transform.location.z
        #             if location_in_region_of_interest(x, y, regions):
        #                 print("position selected!")
        #                 abn_veh_transform = transform
        #                 break
        #         abn_veh_bp = random.choice(blueprints)
        #         try:
        #             print(x, y, z)
        #             abn_veh = world.spawn_actor(abn_veh_bp, abn_veh_transform)
        #             abn_veh.set_autopilot()
        #             abnormals.append(abn_veh)
        #         except:
        #             pass

        #     # for _ in range(50):
        #     #     world.tick()
        #     #     time.sleep(0.05)

        #     for abn_veh in abnormals:
        #         abn_veh.destroy()
        #         print("abnormal vehicle spawning over.")
        # ---------------------------------------

        batch = []
        hero = args.hero
        for n, transform in enumerate(spawn_points):
            print("sencond:", n)
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)  # randomly select a vehicle blueprint
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            light_state = vls.NONE
            if args.car_lights_on:
                light_state = vls.Position | vls.LowBeam | vls.LowBeam

            # spawn the cars and set their autopilot and light state all together
            batch.append(
                SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())).then(
                    SetVehicleLightState(FutureActor, light_state)))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0  # how many pedestrians will run
        percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc is not None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if args.asynch or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        # Note: every tick stands for a simulation step
        while True:
            if not args.asynch and synchronous_master:
                w_frame = world.get_snapshot().frame
                # get vehicle locations
                print("\nWorld's frame: {}".format(w_frame))
                print('\nget vehicle locations')
                sensor_cnt = 0
                for _ in range(10):
                    for _vehicle in vehicles_list:
                        # place sensors in tern on vehicles of interest: pipeline
                        this_vehicle = world.get_actor(_vehicle)
                        loc_dict = {
                            "x": this_vehicle.get_location().x,
                            "y": this_vehicle.get_location().y,
                            "z": this_vehicle.get_location().z
                        }
                        y, x = this_vehicle.get_location().x, this_vehicle.get_location().y  # invert x & y axis in practice
                        if location_in_region_of_interest(x, y, regions):
                            print("Reporting vehicle#{} positions:".format(this_vehicle.id), end=' ---- ')
                            print(this_vehicle.get_location().x, this_vehicle.get_location().y)
                            # ----------------
                            # imu sensor:
                            # ----------------
                            imu_transform = carla.Transform()
                            imu = world.spawn_actor(imu_bp, imu_transform, attach_to=this_vehicle)
                            imu.listen(lambda data: sensor_callback(data, sensor_queue, this_vehicle.id, loc_dict))
                            # ----------------
                            # radar sensor:
                            # ----------------
                            radar_transform = carla.Transform()
                            radar = world.spawn_actor(radar_bp, radar_transform, attach_to=this_vehicle)
                            radar.listen(lambda data: sensor_callback(data, sensor_queue, this_vehicle.id, loc_dict))
                            # ----------------
                            # lidar sensor:
                            # ----------------

                            # ----------------
                            # collision sensor:
                            # ----------------
                            collision_transform = carla.Transform()
                            collision = world.spawn_actor(collision_bp, collision_transform, attach_to=this_vehicle)
                            collision.listen(lambda data: sensor_callback(data, sensor_queue, this_vehicle.id, loc_dict))
                            # ----------------
                            # RSS sensor:
                            # ----------------

                            # ------------- end of sensor placement ------------------

                            world.tick()  # save imu sensor data once
                            # print("#{} vehicle detected in ROI, ticked qsize={}".format(this_vehicle.id, sensor_queue.qsize()))
                            sensor_cnt += 1
                            # destroy each sensor after ticking
                            imu.destroy()
                            radar.destroy()
                            # lidar
                            collision.destroy()
                            # Rss

                            time.sleep(tick_gap)  # wait for sync
                        else:
                            # print("Vehicle#{} not in region: (x, y) = ({}, {})".format(this_vehicle.id, y, x))
                            world.tick()
                            # print("Vehicle not in ROI ticked qsize={}".format(sensor_queue.qsize()))
                        time.sleep(tick_gap)
                try:
                    # for _ in range():
                    no_use_keys = ['raw_data', 'transform', 'frame_number']
                    # imu_df = pd.DataFrame()
                    # radar_df = pd.DataFrame()
                    df_dict = dict()

                    while not sensor_queue.empty():
                        qelem = sensor_queue.get(True, 1.0)
                        frame_num, vehicle_id, sensor_type, src_data, loc_dict = qelem
                        # print(sensor_type)
                        if sensor_type not in df_dict:
                            df_dict[sensor_type] = pd.DataFrame()

                        attrs = [attrib for attrib in dir(src_data) if '__' not in attrib and attrib not in no_use_keys]
                        # print(" frame: #{}\n".format(frame_num), "vehicle: #{}\n".format(vehicle_id),
                        #       "sensor type: {}\n".format(sensor_type), "sensor attributes: {}".format(attrs))
                        merge = {'vehicle_id': vehicle_id}
                        for attrib in attrs:
                            # print("get attrb:", attrib, getattr(src_data, attrib), src_data)
                            data = getattr(src_data, attrib)
                            if type(data).__name__ == 'method':
                                data = data()
                            atom = unfold_array(data, data_name=attrib)
                            merge.update(atom)
                        # print(merge)
                        merge.update(loc_dict)
                        df_dict[sensor_type] = df_dict[sensor_type].append(merge, ignore_index=True)
                        merge.clear()

                    print()
                    for key in df_dict:
                        print(key)
                        df = df_dict[key]
                        print(df)
                        df.to_csv(output_dir + '/{}.csv'.format(key))

                except Empty:
                    print("     Some of the sensor information is missed.")

                break
            else:
                world.wait_for_tick()

    finally:
        # destroying instances to release resources.
        if not args.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying camera')
        # client.apply_batch([carla.command.DestroyActor(camera)])
        camera.destroy()  # this operation causes unexpected issue

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)


if __name__ == '__main__':
    since = time.time()
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
    print("Simulation cost {:.4f} seconds.".format(time.time() - since))

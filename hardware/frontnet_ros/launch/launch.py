import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.conditions import LaunchConfigurationEquals
from launch.conditions import LaunchConfigurationNotEquals
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    # load crazyflies
    crazyflies_yaml = os.path.join(
        get_package_share_directory('frontnet_ros'),
        'config',
        'crazyflies.yaml')

    with open(crazyflies_yaml, 'r') as ymlfile:
        crazyflies = yaml.safe_load(ymlfile)

    # server params
    server_yaml = os.path.join(
        get_package_share_directory('frontnet_ros'),
        'config',
        'server.yaml')

    with open(server_yaml, 'r') as ymlfile:
        server_yaml_contents = yaml.safe_load(ymlfile)

    server_params = [crazyflies] + [server_yaml_contents["/crazyflie_server"]["ros__parameters"]]

    # construct motion_capture_configuration
    motion_capture_yaml = os.path.join(
        get_package_share_directory('frontnet_ros'),
        'config',
        'motion_capture.yaml')

    with open(motion_capture_yaml, 'r') as ymlfile:
        motion_capture = yaml.safe_load(ymlfile)

    motion_capture_params = motion_capture["/motion_capture_tracking"]["ros__parameters"]
    motion_capture_params["rigid_bodies"] = dict()
    for key, value in crazyflies["robots"].items():
        type = crazyflies["robot_types"][value["type"]]
        if value["enabled"] and type["motion_capture"]["enabled"]:
            motion_capture_params["rigid_bodies"][key] =  {
                    "initial_position": value["initial_position"],
                    "marker": type["motion_capture"]["marker"],
                    "dynamics": type["motion_capture"]["dynamics"],
                }

    # copy relevent settings to server params
    server_params[1]["poses_qos_deadline"] = motion_capture_params["topics"]["poses"]["qos"]["deadline"]

    # teleop params
    teleop_params = os.path.join(
        get_package_share_directory('frontnet_ros'),
        'config',
        'teleop.yaml')

    # Rviz config
    rviz_config = os.path.join(
        get_package_share_directory('frontnet_ros'),
        'config',
        'config.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='cpp'),
        DeclareLaunchArgument('debug', default_value='False'),
        Node(
            package='motion_capture_tracking',
            executable='motion_capture_tracking_node',
            condition=LaunchConfigurationNotEquals('backend','sim'),
            name='motion_capture_tracking',
            output='screen',
            parameters=[motion_capture_params]
        ),
        Node(
            package='crazyflie',
            executable='teleop',
            name='teleop',
            remappings=[
                ('emergency', 'all/emergency'),
                ('takeoff', 'all/takeoff'),
                ('land', 'all/land'),
                ('cmd_vel_legacy', 'cf18/cmd_vel_legacy'),
                ('cmd_full_state', 'cf18/cmd_full_state'),
                ('notify_setpoints_stop', 'cf18/notify_setpoints_stop'),
            ],
            parameters=[teleop_params]
        ),
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node', # by default id=0
            # parameters=[{'device_id':1}] # new joystick
        ),
        Node(
            package='crazyflie',
            executable='crazyflie_server.py',
            condition=LaunchConfigurationEquals('backend','cflib'),
            name='crazyflie_server',
            output='screen',
            parameters=server_params
        ),
        Node(
            package='crazyflie',
            executable='crazyflie_server',
            condition=LaunchConfigurationEquals('backend','cpp'),
            name='crazyflie_server',
            output='screen',
            parameters=server_params,
            prefix=PythonExpression(['"xterm -e gdb -ex run --args" if ', LaunchConfiguration('debug'), ' else ""']),
        ),
        Node(
            package='crazyflie_sim',
            executable='crazyflie_server',
            condition=LaunchConfigurationEquals('backend','sim'),
            name='crazyflie_server',
            output='screen',
            emulate_tty=True,
            parameters=server_params
        ),
        # Node(
        #     package='rviz2',
        #     namespace='',
        #     executable='rviz2',
        #     name='rviz2',
        #     arguments=['-d' + rviz_config],
        #     parameters=[{
        #         "use_sim_time": True,
        #     }]
        # ),
        Node(
            package='frontnet_ros',
            namespace='',
            executable='vis2',
            name='vis2',
        ),
        Node(
            package='frontnet_ros',
            namespace='',
            executable='teleop',
            name='teleop2',
        ),
    ])





# import os
# import yaml
# from ament_index_python.packages import get_package_share_directory
# from launch import LaunchDescription
# from launch.actions import DeclareLaunchArgument
# from launch_ros.actions import Node
# from launch.conditions import LaunchConfigurationEquals
# from launch.conditions import LaunchConfigurationNotEquals
# from launch.substitutions import LaunchConfiguration, PythonExpression


# def generate_launch_description():
#     # load crazyflies
#     crazyflies_yaml = os.path.join(
#         get_package_share_directory('frontnet_ros'),
#         'config',
#         'crazyflies.yaml')

#     with open(crazyflies_yaml, 'r') as ymlfile:
#         crazyflies = yaml.safe_load(ymlfile)

#         server_yaml = os.path.join(
#         get_package_share_directory('frontnet_ros'),
#         'config',
#         'server.yaml')

#     with open(server_yaml, 'r') as ymlfile:
#         server_yaml_contents = yaml.safe_load(ymlfile)

#     server_params = [crazyflies] + [server_yaml_contents["/crazyflie_server"]["ros__parameters"]]

#     # construct motion_capture_configuration
#     motion_capture_yaml = os.path.join(
#         get_package_share_directory('frontnet_ros'),
#         'config',
#         'motion_capture.yaml')

#     with open(motion_capture_yaml, 'r') as ymlfile:
#         motion_capture = yaml.safe_load(ymlfile)

#     motion_capture_params = motion_capture["/motion_capture_tracking"]["ros__parameters"]
#     motion_capture_params["rigid_bodies"] = dict()
#     for key, value in crazyflies["robots"].items():
#         type = crazyflies["robot_types"][value["type"]]
#         if value["enabled"] and type["motion_capture"]["enabled"]:
#             motion_capture_params["rigid_bodies"][key] =  {
#                     "initial_position": value["initial_position"],
#                     "marker": type["motion_capture"]["marker"],
#                     "dynamics": type["motion_capture"]["dynamics"],
#                 }

#     # copy relevent settings to server params
#     server_params[1]["poses_qos_deadline"] = motion_capture_params["topics"]["poses"]["qos"]["deadline"]


#     # teleop params
#     teleop_params = os.path.join(
#         get_package_share_directory('frontnet_ros'),
#         'config',
#         'teleop.yaml')

#     # Rviz config
#     rviz_config = os.path.join(
#         get_package_share_directory('frontnet_ros'),
#         'config',
#         'config.rviz')

#     return LaunchDescription([
#         DeclareLaunchArgument('backend', default_value='cpp'),
#         Node(
#             package='motion_capture_tracking',
#             executable='motion_capture_tracking_node',
#             name='motion_capture_tracking',
#             # output='screen',
#             parameters=[motion_capture_params]
#         ),
#         Node(
#             package='crazyflie',
#             executable='teleop',
#             name='teleop',
#             remappings=[
#                 ('emergency', 'all/emergency'),
#                 ('takeoff', 'all/takeoff'),
#                 ('land', 'all/land'),
#                 ('notify_setpoints_stop', 'cf4/notify_setpoints_stop'),
#                 ('cmd_vel', 'cf4/cmd_vel'),
#                 ('cmd_full_state', 'cf4/cmd_full_state'),
#             ],
#             parameters=[teleop_params]
#         ),
#         Node(
#             package='joy',
#             executable='joy_node',
#             name='joy_node' # by default id=0
#         ),
#         Node(
#             package='crazyflie',
#             executable='crazyflie_server.py',
#             condition=LaunchConfigurationEquals('backend','cflib'),
#             name='crazyflie_server',
#             output='screen',
#             parameters=[server_params]
#         ),
#         Node(
#             package='crazyflie',
#             executable='crazyflie_server',
#             condition=LaunchConfigurationEquals('backend','cpp'),
#             name='crazyflie_server',
#             output='screen',
#             parameters=server_params,
#             prefix=PythonExpression(['"xterm -e gdb -ex run --args" if ', LaunchConfiguration('debug'), ' else ""']),
#         ),
#         Node(
#             package='crazyflie_sim',
#             executable='crazyflie_server',
#             condition=LaunchConfigurationEquals('backend','sim'),
#             name='crazyflie_server',
#             output='screen',
#             emulate_tty=True,
#             parameters=[server_params] + [{
#                 "max_dt": 0.1,              # artificially limit the step() function (set to 0 to disable)
#             }]
#         ),
#         Node(
#             package='rviz2',
#             namespace='',
#             executable='rviz2',
#             name='rviz2',
#             arguments=['-d' + rviz_config]
#         ),
#         Node(
#             package='frontnet_ros',
#             namespace='',
#             executable='vis2',
#             name='vis2',
#         ),
#         Node(
#             package='frontnet_ros',
#             namespace='',
#             executable='teleop',
#             name='teleop2',
#         ),
#     ])

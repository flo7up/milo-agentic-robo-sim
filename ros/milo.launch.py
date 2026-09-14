from pathlib import Path
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    root = Path(__file__).resolve().parent
    bringup = Path(get_package_share_directory("nav2_bringup"))
    params = yaml.safe_load((bringup / "params/nav2_params.yaml").read_text())
    overrides = yaml.safe_load((root / "nav2.yaml").read_text())

    def merge(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(params, overrides)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as output:
        yaml.safe_dump(params, output)
        config = output.name
    bridge = ExecuteProcess(cmd=["python3", str(root / "bridge.py"), "--backend", LaunchConfiguration("backend")], output="screen")
    return LaunchDescription([
        DeclareLaunchArgument("backend", default_value="http://host.docker.internal:8012"),
        bridge,
        RegisterEventHandler(OnProcessExit(target_action=bridge, on_exit=[Shutdown(reason="Milo bridge ended")])),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(bringup / "launch/navigation_launch.py")),
            launch_arguments={"params_file": config, "use_sim_time": "True", "autostart": "True", "use_composition": "False"}.items())])
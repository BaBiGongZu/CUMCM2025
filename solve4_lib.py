"""
问题四：利用FY1、FY2、FY3等3架无人机各投放1枚烟幕干扰弹实施对M1的干扰
使用C库加速计算的优化程序
"""

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import time
import warnings
import math
import ctypes
import os
import platform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- C库加载器 (与solve3_lib.py相同) ---
class MultiCloudSmokeBlockingLib:
    def __init__(self, lib_dir="libs"):
        self.lib_dir = lib_dir
        self.c_lib = None
        self._load_library()
        self._setup_function_interfaces()
        self._verify_library()

    def _detect_platform(self):
        system = platform.system().lower()
        if system == 'darwin': return 'macos', '.dylib'
        if system == 'linux': return 'linux', '.so'
        if system == 'windows': return 'windows', '.dll'
        raise RuntimeError(f"不支持的平台: {system}")

    def _load_library(self):
        platform_name, extension = self._detect_platform()
        lib_path = os.path.join(self.lib_dir, f"smoke_blocking_{platform_name}{extension}")
        if not os.path.exists(lib_path):
             lib_path = os.path.join(self.lib_dir, f"smoke_blocking{extension}")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"找不到动态库。请先编译C库。查找路径: {lib_path}")

        try:
            self.c_lib = ctypes.CDLL(lib_path)
            print(f"✅ 成功加载C库: {lib_path}")
        except OSError as e:
            raise OSError(f"加载C库失败: {e}")

    def _setup_function_interfaces(self):
        """设置C函数接口"""
        self.c_lib.calculate_total_blocking_duration.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # missile_start
            ctypes.POINTER(ctypes.c_double),  # missile_velocity
            ctypes.POINTER(ctypes.c_double),  # explode_positions_flat
            ctypes.POINTER(ctypes.c_double),  # explode_times
            ctypes.c_int,                     # num_clouds
            ctypes.c_double,                  # total_flight_time
            ctypes.c_double,                  # time_step
            ctypes.c_double,                  # sink_speed
            ctypes.c_double                   # smoke_duration
        ]
        self.c_lib.calculate_total_blocking_duration.restype = ctypes.c_double

        self.c_lib.get_version.restype = ctypes.c_char_p
        self.c_lib.get_key_points_count.restype = ctypes.c_int
        self.c_lib.get_sampling_info.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]

    def _verify_library(self):
        version = self.get_version()
        key_points_count = self.get_key_points_count()
        total, bottom, top, side = (ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int())
        self.c_lib.get_sampling_info(ctypes.byref(total), ctypes.byref(bottom), ctypes.byref(top), ctypes.byref(side))
        print(f"C库版本: {version}")
        print(f"关键点总数: {key_points_count}")
        print(f"采样分布: 底面{bottom.value} + 顶面{top.value} + 侧面{side.value} = {total.value}")

    def get_version(self):
        return self.c_lib.get_version().decode('utf-8')

    def get_key_points_count(self):
        """获取C库中定义的关键点数量"""
        return self.c_lib.get_key_points_count()

    def calculate_total_duration(self, missile_start, missile_velocity,
                                 explode_positions, explode_times, total_flight_time,
                                 time_step, sink_speed, smoke_duration):
        """调用C核心函数"""
        explode_positions = np.array(explode_positions, dtype=np.float64)
        explode_times = np.array(explode_times, dtype=np.float64)
        num_clouds = len(explode_positions)

        missile_start_c = missile_start.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        missile_velocity_c = missile_velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        explode_pos_flat = explode_positions.flatten()
        explode_pos_c = explode_pos_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        explode_times_c = explode_times.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        return self.c_lib.calculate_total_blocking_duration(
            missile_start_c, missile_velocity_c,
            explode_pos_c, explode_times_c,
            num_clouds, total_flight_time, time_step, sink_speed, smoke_duration
        )

# --- 全局常量 ---
G_ACCEL = 9.8
P_M1_0 = np.array([20000.0, 0.0, 2000.0], dtype=np.float64)
V_M1_SPEED = 300.0
TARGET_FALSE = np.array([0.0, 0.0, 0.0], dtype=np.float64)
VEC_M1_TO_TARGET = TARGET_FALSE - P_M1_0
DIST_M1_TO_TARGET = np.linalg.norm(VEC_M1_TO_TARGET)
UNIT_VEC_M1 = VEC_M1_TO_TARGET / DIST_M1_TO_TARGET
VEC_V_M1 = np.array(V_M1_SPEED * UNIT_VEC_M1, dtype=np.float64)
MISSILE_FLIGHT_TIME = DIST_M1_TO_TARGET / V_M1_SPEED

# 三架无人机的初始位置
P_FY1_0 = np.array([17800.0, 0.0, 1800.0], dtype=np.float64)
P_FY2_0 = np.array([12000.0, 1400.0, 1400.0], dtype=np.float64)
P_FY3_0 = np.array([6000.0, -3000.0, 700.0], dtype=np.float64)
UAV_POSITIONS = [P_FY1_0, P_FY2_0, P_FY3_0]

REAL_TARGET_HEIGHT = 10.0
SMOKE_DURATION = 20.0
OPTIMIZER_TIME_STEP = 0.01
NUM_UAVS = 3
V_SMOKE_SINK_SPEED = 3.0

# 初始化C库
print("🔧 初始化三架无人机烟幕弹C库...")
c_smoke_lib = MultiCloudSmokeBlockingLib()

def calculate_uav_direction_from_angle(flight_angle):
    """根据飞行角度计算无人机方向向量"""
    uav_direction = np.zeros(3, dtype=np.float64)
    uav_direction[0] = np.cos(flight_angle)
    uav_direction[1] = np.sin(flight_angle)
    return uav_direction

def decode_params_to_trajectories(params):
    """
    解码参数为轨迹信息
    参数顺序: [speed1, angle1, t_drop1, t_delay1, speed2, angle2, t_drop2, t_delay2, speed3, angle3, t_drop3, t_delay3]
    """
    trajectories = []

    for i in range(NUM_UAVS):
        uav_speed = params[i * 4]
        flight_angle = params[i * 4 + 1]
        t_drop = params[i * 4 + 2]
        t_delay = params[i * 4 + 3]

        uav_direction = calculate_uav_direction_from_angle(flight_angle)
        uav_initial_pos = UAV_POSITIONS[i]

        # 计算投放位置
        drop_pos = uav_initial_pos + uav_speed * t_drop * uav_direction

        # 计算起爆位置
        bomb_initial_velocity = uav_speed * uav_direction
        explode_pos = drop_pos + bomb_initial_velocity * t_delay
        explode_pos[2] -= 0.5 * G_ACCEL * t_delay**2

        t_explode = t_drop + t_delay

        trajectories.append({
            'uav_id': i + 1,
            'uav_pos': uav_initial_pos.copy(),
            'uav_speed': uav_speed,
            'flight_angle': flight_angle,
            'drop_pos': drop_pos.copy(),
            'explode_pos': explode_pos.copy(),
            't_drop': t_drop,
            't_delay': t_delay,
            't_explode': t_explode
        })

    return trajectories

def three_uavs_objective_function(params):
    """三架无人机的C库加速目标函数"""
    try:
        trajectories = decode_params_to_trajectories(params)

        explode_positions = []
        explode_times = []

        for traj in trajectories:
            # 约束检查
            if traj['t_explode'] >= MISSILE_FLIGHT_TIME or traj['explode_pos'][2] <= REAL_TARGET_HEIGHT:
                return 0.0
            explode_positions.append(traj['explode_pos'])
            explode_times.append(traj['t_explode'])

        # 调用C库计算总遮蔽时长
        duration = c_smoke_lib.calculate_total_duration(
            P_M1_0, VEC_V_M1,
            explode_positions, explode_times,
            MISSILE_FLIGHT_TIME, OPTIMIZER_TIME_STEP,
            V_SMOKE_SINK_SPEED, SMOKE_DURATION
        )

        return -duration

    except Exception:
        return 0.0

def print_solution_details(params, duration):
    """打印解的详细信息"""
    trajectories = decode_params_to_trajectories(params)

    print(f"\n找到最优策略:")
    print(f"  > 最大有效遮蔽时长: {duration:.6f} 秒")
    print("=" * 80)

    for traj in trajectories:
        print(f"无人机 FY{traj['uav_id']}:")
        print(f"  初始位置: [{traj['uav_pos'][0]:.1f}, {traj['uav_pos'][1]:.1f}, {traj['uav_pos'][2]:.1f}]")
        print(f"  飞行参数: 速度 {traj['uav_speed']:.2f} m/s, 角度 {math.degrees(traj['flight_angle']):.2f}°")
        print(f"  投放时间: {traj['t_drop']:.3f} s")
        print(f"  延迟时间: {traj['t_delay']:.3f} s")
        print(f"  起爆时间: {traj['t_explode']:.3f} s")
        print(f"  投放位置: [{traj['drop_pos'][0]:.1f}, {traj['drop_pos'][1]:.1f}, {traj['drop_pos'][2]:.1f}]")
        print(f"  起爆位置: [{traj['explode_pos'][0]:.1f}, {traj['explode_pos'][1]:.1f}, {traj['explode_pos'][2]:.1f}]")
        print()
    print("=" * 80)

def visualize_solution(params):
    """可视化解决方案"""
    trajectories = decode_params_to_trajectories(params)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制导弹轨迹
    missile_trajectory_x = [P_M1_0[0], TARGET_FALSE[0]]
    missile_trajectory_y = [P_M1_0[1], TARGET_FALSE[1]]
    missile_trajectory_z = [P_M1_0[2], TARGET_FALSE[2]]
    ax.plot(missile_trajectory_x, missile_trajectory_y, missile_trajectory_z,
            'r-', linewidth=3, label='导弹轨迹', alpha=0.8)

    # 绘制目标
    ax.scatter(*TARGET_FALSE, color='red', s=200, marker='*', label='假目标')
    ax.scatter(*[0, 200, 0], color='green', s=200, marker='s', label='真目标')

    colors = ['blue', 'orange', 'purple']
    for i, traj in enumerate(trajectories):
        color = colors[i]

        # 无人机初始位置
        ax.scatter(*traj['uav_pos'], color=color, s=100, marker='^',
                  label=f'FY{traj["uav_id"]} 初始位置')

        # 投放位置
        ax.scatter(*traj['drop_pos'], color=color, s=80, marker='o', alpha=0.7)

        # 起爆位置
        ax.scatter(*traj['explode_pos'], color=color, s=150, marker='*', alpha=0.8,
                  label=f'FY{traj["uav_id"]} 起爆点')

        # 无人机飞行轨迹
        uav_direction = calculate_uav_direction_from_angle(traj['flight_angle'])
        end_pos = traj['uav_pos'] + traj['uav_speed'] * traj['t_drop'] * uav_direction
        ax.plot([traj['uav_pos'][0], end_pos[0]],
                [traj['uav_pos'][1], end_pos[1]],
                [traj['uav_pos'][2], end_pos[2]],
                color=color, linestyle='--', alpha=0.6)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.set_title('三架无人机烟幕弹投放策略')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🎯 问题四：三架无人机协同烟幕干扰优化")
    print(f"导弹总飞行时间: {MISSILE_FLIGHT_TIME:.2f} s")
    print("无人机位置:")
    for i, pos in enumerate(UAV_POSITIONS):
        print(f"  FY{i+1}: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")

    # 参数边界: [speed1, angle1, t_drop1, t_delay1, speed2, angle2, t_drop2, t_delay2, speed3, angle3, t_drop3, t_delay3]
    bounds = [
        # FY1 参数
        (70.0, 140.0),      # speed1
        (0, 2 * np.pi),     # angle1
        (0.0, 15.0),        # t_drop1
        (0.0, 20.0),        # t_delay1
        # FY2 参数
        (70.0, 140.0),      # speed2
        (0, 2 * np.pi),     # angle2
        (0.0, 15.0),        # t_drop2
        (0.0, 20.0),        # t_delay2
        # FY3 参数
        (70.0, 140.0),      # speed3
        (0, 2 * np.pi),     # angle3
        (0.0, 15.0),        # t_drop3
        (0.0, 20.0),        # t_delay3
    ]

    # 启发式种子（基于问题三的经验）
    heuristic_seeds = [
        [130.0, 3.14, 2.0, 5.0, 120.0, 1.57, 3.0, 6.0, 110.0, 4.71, 4.0, 7.0],
        [140.0, 0.0, 1.0, 4.0, 100.0, 2.35, 2.5, 5.5, 115.0, 5.50, 3.5, 6.5]
    ]

    TOTAL_POPSIZE = 800
    num_seeds = len(heuristic_seeds)
    num_random = TOTAL_POPSIZE - num_seeds

    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    sampler = qmc.LatinHypercube(d=len(bounds), seed=42)
    random_init_unit_scale = sampler.random(n=num_random)
    scaled_random_init = qmc.scale(random_init_unit_scale, lower_bounds, upper_bounds)

    initial_population = np.vstack([heuristic_seeds, scaled_random_init])

    print(f"初始种群大小: {TOTAL_POPSIZE} (含{num_seeds}个启发式种子)")
    print("开始优化...")

    start_time = time.time()

    result = differential_evolution(
        three_uavs_objective_function,
        bounds,
        init=initial_population,
        strategy='rand1bin',
        maxiter=3000,
        popsize=150,
        tol=0.001,
        recombination=0.7,
        mutation=(0.5, 1.0),
        disp=True,
        workers=-1,
        seed=42
    )

    end_time = time.time()
    print(f"\n优化完成，总耗时: {end_time - start_time:.2f} 秒")

    if result.success and result.fun < -1e-6:
        best_params = result.x
        max_duration = -result.fun
        print(f"\n🎯 优化收敛成功!")
        print_solution_details(best_params, max_duration)

        print("\n" + "="*25 + " 参数输出 " + "="*25)
        seed_string = ", ".join([f"{val:.6f}" for val in best_params])
        print(f"[{seed_string}]")
        print("="*70)

        # 可视化结果
        print("\n生成可视化图像...")
        visualize_solution(best_params)

    else:
        print(f"\n❌ 优化未收敛或未找到有效解")
        if hasattr(result, 'x'):
            print("\n" + "="*25 + " 当前最佳参数 " + "="*25)
            seed_string = ", ".join([f"{val:.6f}" for val in result.x])
            print(f"[{seed_string}]")
            print("="*70)


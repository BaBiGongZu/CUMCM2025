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

# 新增：将 mealpy 相关的导入移至文件顶部
from mealpy.evolutionary_based.DE import JADE
import mealpy as mp
from mealpy.utils.space import FloatVar

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
OPTIMIZER_TIME_STEP = 0.05
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

# ==============================================================================
# 新增：定义一个符合 mealpy 3.x 规范的 Problem 子类
# ==============================================================================
class UAVProblem(mp.Problem):
    """
    一个继承自 mealpy.Problem 的自定义问题类。
    """
    def __init__(self, bounds, minmax="min", **kwargs):
        """
        在初始化时，我们设置好边界和优化方向。
        """
        super().__init__(bounds=bounds, minmax=minmax, **kwargs)

    def obj_func(self, solution):
        """
        实现这个方法是关键。
        mealpy 的优化器会调用这个方法来计算每个解的适应度。
        我们在这里调用我们已经写好的目标函数。
        """
        return three_uavs_objective_function(solution)


# ==============================================================================
# 优化主流程 (v3.9 - 最终面向对象版)
# ==============================================================================
if __name__ == "__main__":
    print("🎯 问题四：三架无人机协同烟幕干扰优化 (使用JADE算法)")
    print(f"导弹总飞行时间: {MISSILE_FLIGHT_TIME:.2f} s")

    # 1. 定义参数边界 (使用 FloatVar)
    # 这是 mealpy 3.x 的标准做法，为每个变量创建一个 FloatVar 对象
    bounds_list = [
        # UAV 1: speed, angle, t_drop, t_delay
        FloatVar(lb=70.0, ub=140.0),
        FloatVar(lb=0.0, ub=2 * np.pi),
        FloatVar(lb=0.0, ub=60.0),
        FloatVar(lb=0.0, ub=20.0),
        # UAV 2: speed, angle, t_drop, t_delay
        FloatVar(lb=70.0, ub=140.0),
        FloatVar(lb=0.0, ub=2 * np.pi),
        FloatVar(lb=0.0, ub=60.0),
        FloatVar(lb=0.0, ub=20.0),
        # UAV 3: speed, angle, t_drop, t_delay
        FloatVar(lb=70.0, ub=140.0),
        FloatVar(lb=0.0, ub=2 * np.pi),
        FloatVar(lb=0.0, ub=60.0),
        FloatVar(lb=0.0, ub=20.0),
    ]

    # 2. 实例化我们自定义的 UAVProblem 类
    problem = UAVProblem(bounds=bounds_list, minmax="min", verbose=True)

    # 3. 设置 JADE 算法的超参数
    epoch = 30000
    pop_size = 500
    
    # 实例化JADE模型
    model = JADE(epoch=epoch, pop_size=pop_size)

    print("\n开始使用JADE算法进行优化...")
    start_time = time.time()

    # 4. 运行求解器，传入我们自定义的 problem 对象
    best_params, best_fitness = model.solve(problem)

    end_time = time.time()
    print(f"\n优化完成，总耗时: {end_time - start_time:.2f} 秒")

    # 5. 输出结果
    if best_fitness < -1e-6:
        max_duration = -best_fitness
        print(f"\n🎯 JADE算法优化收敛成功!")
        print_solution_details(best_params, max_duration)

        print("\n" + "="*25 + " 参数输出 " + "="*25)
        seed_string = ", ".join([f"{val:.6f}" for val in best_params])
        print(f"[{seed_string}]")
        print("="*70)

        print("\n生成可视化图像...")
        visualize_solution(best_params)
    else:
        print(f"\n❌ JADE算法未找到有效解")
        if best_params is not None:
            print("\n" + "="*25 + " 当前最佳参数 " + "="*25)
            seed_string = ", ".join([f"{val:.6f}" for val in best_params])
            print(f"[{seed_string}]")
            print("="*70)


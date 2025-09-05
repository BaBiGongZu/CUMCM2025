"""
问题三：利用无人机FY1投放3枚烟幕干扰弹实施对M1的干扰
使用C库加速计算的优化程序 (V2 - 重构版)
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

# --- C库加载器 (已更新以适应新函数) ---
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
        """设置C函数接口 (已更新)"""
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
        """调用新的C核心函数"""
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
P_FY1_0 = np.array([17800.0, 0.0, 1800.0], dtype=np.float64)
REAL_TARGET_HEIGHT = 10.0
SMOKE_DURATION = 20.0
OPTIMIZER_TIME_STEP = 0.01
NUM_SMOKE_BOMBS = 3
MIN_DROP_INTERVAL = 1.0
V_SMOKE_SINK_SPEED = 3.0

# 初始化C库
print("🔧 初始化多烟雾弹C库 (V2)...")
c_smoke_lib = MultiCloudSmokeBlockingLib()

def calculate_uav_direction_from_angle(flight_angle):
    uav_direction = np.zeros(3, dtype=np.float64)
    uav_direction[0] = np.cos(flight_angle)
    uav_direction[1] = np.sin(flight_angle)
    return uav_direction

def decode_params_to_trajectories(params):
    uav_speed, flight_angle, t_drop1, t_interval2, t_interval3, t_delay1, t_delay2, t_delay3 = params
    uav_direction = calculate_uav_direction_from_angle(flight_angle)
    
    absolute_drop_times = [t_drop1, t_drop1 + t_interval2, t_drop1 + t_interval2 + t_interval3]
    delays = [t_delay1, t_delay2, t_delay3]
    
    trajectories = []
    for i in range(NUM_SMOKE_BOMBS):
        t_drop = absolute_drop_times[i]
        t_delay = delays[i]
        drop_pos = P_FY1_0 + uav_speed * t_drop * uav_direction
        bomb_initial_velocity = uav_speed * uav_direction
        explode_pos = drop_pos + bomb_initial_velocity * t_delay
        explode_pos[2] -= 0.5 * G_ACCEL * t_delay**2
        trajectories.append({'explode_pos': explode_pos, 't_explode': t_drop + t_delay})
    
    return trajectories

def three_bombs_objective_function(params):
    """三枚烟幕干扰弹的C库加速目标函数 (V2 - 简化版)"""
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
        
        # 一次性调用C库计算总时长
        duration = c_smoke_lib.calculate_total_duration(
            P_M1_0, VEC_V_M1,
            explode_positions, explode_times,
            MISSILE_FLIGHT_TIME, OPTIMIZER_TIME_STEP,
            V_SMOKE_SINK_SPEED, SMOKE_DURATION
        )
        
        return -duration
        
    except Exception:
        return 0.0

# --- 结果打印和验证函数 (保持不变，但现在基于正确的结果) ---
def print_solution_details(params, duration):
    # ... (此函数无需修改，它调用decode_params_to_trajectories，现在将打印正确的结果)
    uav_speed, flight_angle, t_drop1, t_interval2, t_interval3, t_delay1, t_delay2, t_delay3 = params
    uav_direction = calculate_uav_direction_from_angle(flight_angle)
    absolute_drop_times = [t_drop1, t_drop1 + t_interval2, t_drop1 + t_interval2 + t_interval3]
    delays = [t_delay1, t_delay2, t_delay3]
    
    print(f"\n找到最优策略 (V2):")
    print(f"  > 最大有效遮蔽时长: {duration:.6f} 秒")
    print("=" * 80)
    print(f"无人机参数: 速度 {uav_speed:.2f} m/s, 角度 {math.degrees(flight_angle):.2f}°")
    
    for i in range(NUM_SMOKE_BOMBS):
        t_drop = absolute_drop_times[i]
        t_delay = delays[i]
        t_explode = t_drop + t_delay
        drop_pos = P_FY1_0 + uav_speed * t_drop * uav_direction
        bomb_initial_velocity = uav_speed * uav_direction
        explode_pos = drop_pos + bomb_initial_velocity * t_delay
        explode_pos[2] -= 0.5 * G_ACCEL * t_delay**2
        print(f"  烟幕弹 {i+1}: 投放 {t_drop:.3f}s, 延迟 {t_delay:.3f}s, 起爆 {t_explode:.3f}s at [{explode_pos[0]:.1f}, {explode_pos[1]:.1f}, {explode_pos[2]:.1f}]")
    print("=" * 80)

if __name__ == "__main__":
    print("🎯 问题三：C库加速优化程序 (V2 - 重构版)")
    print(f"导弹总飞行时间: {MISSILE_FLIGHT_TIME:.2f} s")
    
    bounds = [
        (70.0, 140.0),      # uav_speed
        (0, 2 * np.pi),     # flight_angle
        (0.0, 10.0),        # t_drop1
        (MIN_DROP_INTERVAL, 20.0), # t_interval2
        (MIN_DROP_INTERVAL, 20.0), # t_interval3
        (0.0, 20.0),        # t_delay1
        (0.0, 20.0),        # t_delay2
        (0.0, 20.0),        # t_delay3
    ]
    
    heuristic_seeds = [
        [139.976844, 3.135471, 0.001625, 3.656653, 1.919484, 3.609524, 5.341602, 6.039781]
    ]
    
    TOTAL_POPSIZE = 1000
    num_seeds = len(heuristic_seeds)
    num_random = TOTAL_POPSIZE - num_seeds
    
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    sampler = qmc.LatinHypercube(d=len(bounds), seed=1)
    random_init_unit_scale = sampler.random(n=num_random)
    scaled_random_init = qmc.scale(random_init_unit_scale, lower_bounds, upper_bounds)
    
    initial_population = np.vstack([heuristic_seeds, scaled_random_init])
    
    print(f"初始种群大小: {TOTAL_POPSIZE} (含{num_seeds}个种子)")
    print("开始优化...")
    
    start_time = time.time()
    
    result = differential_evolution(
        three_bombs_objective_function,
        bounds,
        init=initial_population,
        strategy='best1bin', # 使用 'best1bin' 策略以利用种子
        maxiter=5000,
        popsize=200,
        tol=0.001,
        recombination=0.7,
        mutation=(0.7, 1.0),
        disp=True,
        workers=-1,
        seed=4
    )
    
    end_time = time.time()
    print(f"\n优化完成，总耗时: {end_time - start_time:.2f} 秒")
    
    if result.success and result.fun < -1e-6:
        best_params = result.x
        max_duration = -result.fun
        print(f"\n🎯 优化收敛成功!")
        print_solution_details(best_params, max_duration)
        
        print("\n" + "="*25 + " 下一轮迭代种子 " + "="*25)
        seed_string = ", ".join([f"{val:.6f}" for val in best_params])
        print(f"[{seed_string}]")
        print("="*70)
    else:
        print(f"\n❌ 优化未收敛或未找到有效解")
        if hasattr(result, 'x'):
            print("\n" + "="*25 + " 当前最佳种子 " + "="*25)
            seed_string = ", ".join([f"{val:.6f}" for val in result.x])
            print(f"[{seed_string}]")
            print("="*70)

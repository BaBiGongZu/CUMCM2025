"""
求解无人机FY2和FY3单独的最优策略
参数与solve4_lib.py保持一致
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

# --- C库加载器 (与solve2_lib.py相同) ---
class SmokeBlockingCLib:
    def __init__(self, lib_dir="libs"):
        self.lib_dir = lib_dir
        self.c_lib = None
        self._load_library()
        self._setup_function_interfaces()
        self._verify_library()

    def _detect_platform(self):
        """检测当前平台"""
        system = platform.system().lower()

        if system == 'darwin':
            return 'macos', '.dylib'
        elif system == 'linux':
            return 'linux', '.so'
        elif system == 'windows':
            return 'windows', '.dll'
        else:
            raise RuntimeError(f"不支持的平台: {system}")

    def _load_library(self):
        """加载动态库"""
        platform_name, extension = self._detect_platform()

        # 尝试多个可能的路径
        possible_paths = [
            os.path.join(self.lib_dir, f"smoke_blocking_{platform_name}{extension}"),
            os.path.join(self.lib_dir, f"smoke_blocking{extension}"),
            f"smoke_blocking_{platform_name}{extension}",
            f"smoke_blocking{extension}",
        ]

        for lib_path in possible_paths:
            if os.path.exists(lib_path):
                try:
                    self.c_lib = ctypes.CDLL(lib_path)
                    print(f"✅ 成功加载C库: {lib_path}")
                    return
                except OSError as e:
                    print(f"⚠️  尝试加载失败: {lib_path} - {e}")
                    continue

        raise FileNotFoundError(
            f"找不到动态库。请先编译C库。\n"
            f"查找路径: {possible_paths}"
        )

    def _setup_function_interfaces(self):
        """设置C函数接口"""
        # 新的核心函数 - calculate_total_blocking_duration
        self.c_lib.calculate_total_blocking_duration.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # missile_start_arr
            ctypes.POINTER(ctypes.c_double),  # missile_velocity_arr
            ctypes.POINTER(ctypes.c_double),  # explode_positions_flat
            ctypes.POINTER(ctypes.c_double),  # explode_times_arr
            ctypes.c_int,                     # num_clouds
            ctypes.c_double,                  # total_flight_time
            ctypes.c_double,                  # time_step
            ctypes.c_double,                  # sink_speed
            ctypes.c_double                   # smoke_duration
        ]
        self.c_lib.calculate_total_blocking_duration.restype = ctypes.c_double

        # 其他辅助函数
        self.c_lib.get_version.restype = ctypes.c_char_p
        self.c_lib.get_key_points_count.restype = ctypes.c_int

        self.c_lib.get_sampling_info.argtypes = [
            ctypes.POINTER(ctypes.c_int),     # total
            ctypes.POINTER(ctypes.c_int),     # bottom
            ctypes.POINTER(ctypes.c_int),     # top
            ctypes.POINTER(ctypes.c_int)      # side
        ]

    def _verify_library(self):
        """验证库的正确性"""
        version = self.get_version()
        key_points_count = self.get_key_points_count()

        total = ctypes.c_int()
        bottom = ctypes.c_int()
        top = ctypes.c_int()
        side = ctypes.c_int()
        self.c_lib.get_sampling_info(
            ctypes.byref(total), ctypes.byref(bottom),
            ctypes.byref(top), ctypes.byref(side)
        )

        print(f"C库版本: {version}")
        print(f"关键点数量: {key_points_count}")
        print(f"采样分布: 底面{bottom.value} + 顶面{top.value} + 侧面{side.value} = {total.value}")

        if key_points_count != total.value:
            raise RuntimeError(f"关键点数量不一致: {key_points_count} vs {total.value}")

    def get_version(self):
        """获取库版本"""
        return self.c_lib.get_version().decode('utf-8')

    def get_key_points_count(self):
        """获取关键点数量"""
        return self.c_lib.get_key_points_count()

    def calculate_total_duration(self, missile_start, missile_velocity,
                                explode_positions, explode_times, total_flight_time,
                                time_step, sink_speed, smoke_duration):
        """计算总遮蔽时长"""
        missile_start = np.array(missile_start, dtype=np.float64)
        missile_velocity = np.array(missile_velocity, dtype=np.float64)
        explode_positions = np.array(explode_positions, dtype=np.float64)
        explode_times = np.array(explode_times, dtype=np.float64)

        num_clouds = len(explode_positions)

        missile_start_c = missile_start.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        missile_velocity_c = missile_velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        explode_pos_flat = explode_positions.flatten()
        explode_pos_c = explode_pos_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        explode_times_c = explode_times.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        return self.c_lib.calculate_total_blocking_duration(
            missile_start_c,
            missile_velocity_c,
            explode_pos_c,
            explode_times_c,
            num_clouds,
            total_flight_time,
            time_step,
            sink_speed,
            smoke_duration
        )

# --- 全局常量（与solve4_lib.py保持一致）---
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

REAL_TARGET_HEIGHT = 10.0
SMOKE_DURATION = 20.0
OPTIMIZER_TIME_STEP = 0.1
V_SMOKE_SINK_SPEED = 3.0

# 初始化C库
print("🔧 初始化C库...")
c_smoke_lib = SmokeBlockingCLib()

# 全局变量用于目标函数
current_uav_pos = None
current_uav_name = None

def calculate_uav_direction_from_angle(flight_angle):
    """根据飞行角度计算无人机飞行方向"""
    uav_direction = np.zeros(3)
    uav_direction[0] = np.cos(flight_angle)
    uav_direction[1] = np.sin(flight_angle)
    return uav_direction

def objective_function(params):
    """
    使用C库加速的目标函数（模块级别函数，支持多进程）

    Parameters:
    -----------
    params : array-like
        [uav_speed, flight_angle, t_drop, t_explode_delay]
    """
    uav_speed, flight_angle, t_drop, t_explode_delay = params

    # 计算总起爆时间
    t_explode_abs = t_drop + t_explode_delay

    # 如果起爆时间超过导弹飞行时间，无效
    if t_explode_abs >= MISSILE_FLIGHT_TIME:
        # print(f"⚠️  {current_uav_name} 起爆时间超出导弹飞行时间，跳过")
        return 0.0

    # 1. 计算无人机飞行方向
    uav_direction = calculate_uav_direction_from_angle(flight_angle)

    # 2. 计算烟雾弹投放位置
    uav_drop_pos = current_uav_pos + uav_speed * t_drop * uav_direction

    # 3. 计算烟雾弹起爆位置
    bomb_initial_velocity = uav_speed * uav_direction
    explode_pos = uav_drop_pos.copy()
    explode_pos += bomb_initial_velocity * t_explode_delay
    explode_pos[2] -= 0.5 * G_ACCEL * t_explode_delay**2

    # print(explode_pos)
    # 检查起爆高度是否合理
    if explode_pos[2] <= REAL_TARGET_HEIGHT:
        # print(f"⚠️  {current_uav_name} 烟雾弹在地面以下起爆，跳过")
        return 0.0

    # 4. 使用C库接口进行计算
    explode_positions = [explode_pos]  # 单个烟雾弹
    explode_times = [t_explode_abs]

    # 调用C库函数
    total_duration = c_smoke_lib.calculate_total_duration(
        P_M1_0, VEC_V_M1, explode_positions, explode_times,
        MISSILE_FLIGHT_TIME, OPTIMIZER_TIME_STEP,
        V_SMOKE_SINK_SPEED, SMOKE_DURATION
    )

    return -total_duration  # 负值用于最大化

# --- 优化与求解 ---
def print_solution_details(params, duration, uav_name, uav_initial_pos):
    """打印解的详细信息"""
    uav_speed, flight_angle, t_drop, t_explode_delay = params
    t_explode_abs = t_drop + t_explode_delay

    uav_direction = calculate_uav_direction_from_angle(flight_angle)
    uav_drop_pos = uav_initial_pos + uav_speed * t_drop * uav_direction
    bomb_initial_velocity = uav_speed * uav_direction
    explode_pos = uav_drop_pos.copy()
    explode_pos += bomb_initial_velocity * t_explode_delay
    explode_pos[2] -= 0.5 * G_ACCEL * t_explode_delay**2

    print(f"\n找到{uav_name}最优策略：")
    print(f"  > 最大有效遮蔽时长: {duration:.6f} 秒")
    print("-" * 60)
    print(f"  无人机飞行速度: {uav_speed:.4f} m/s")
    print(f"  飞行角度: {flight_angle:.6f} 弧度 ({math.degrees(flight_angle):.2f}°)")
    print(f"  受领任务后投放时间: {t_drop:.4f} s")
    print(f"  投放后起爆延迟: {t_explode_delay:.4f} s")
    print(f"  总起爆时间: {t_explode_abs:.4f} s")
    print("-" * 60)
    print(f"  无人机初始位置: [{uav_initial_pos[0]:.1f}, {uav_initial_pos[1]:.1f}, {uav_initial_pos[2]:.1f}]")
    print(f"  无人机飞行方向: [{uav_direction[0]:.6f}, {uav_direction[1]:.6f}, {uav_direction[2]:.6f}]")
    print(f"  投放位置: [{uav_drop_pos[0]:.2f}, {uav_drop_pos[1]:.2f}, {uav_drop_pos[2]:.2f}]")
    print(f"  起爆位置: [{explode_pos[0]:.2f}, {explode_pos[1]:.2f}, {explode_pos[2]:.2f}]")
    print("-" * 60)

def optimize_single_uav(uav_name, uav_initial_pos):
    """优化单个无人机的策略"""
    global current_uav_pos, current_uav_name

    print(f"\n🎯 开始优化{uav_name}单独策略")
    print("=" * 60)
    print(f"{uav_name}初始位置: [{uav_initial_pos[0]:.1f}, {uav_initial_pos[1]:.1f}, {uav_initial_pos[2]:.1f}]")

    # 设置全局变量
    current_uav_pos = uav_initial_pos.copy()
    current_uav_name = uav_name

    # 根据无人机调整优化变量边界
    if uav_name == "FY2":
        # FY2的标准边界
        bounds = [
            (70.0, 140.0),      # uav_speed (m/s)
            (0, 2 * np.pi),     # flight_angle (rad)
            (0.0, 15.0),        # t_drop (s)
            (0.0, 20.0)         # t_explode_delay (s)
        ]
    else:  # FY3
        # FY3需要特殊处理：缩短延迟时间，减少地面引爆概率
        bounds = [
            (70.0, 140.0),      # uav_speed (m/s) - 保持不变
            (0, 2 * np.pi),     # flight_angle (rad) - 保持不变
            (0.0, 10.0),        # t_drop (s) - 缩短投放时间
            (0.0, 8.0)          # t_explode_delay (s) - 大幅缩短延迟时间
        ]

    # 重新设计启发式种子 - 基于几何分析选择更合理的参数
    if uav_name == "FY2":
        # FY2位置: [12000, 1400, 1400]，需要向导弹轨迹前方投放
        heuristic_seeds = [
            [90.0, 4.0, 2.0, 15.0],     # 向西北方向，长延迟
            [100.0, 3.5, 3.0, 12.0],    # 向西方向
            [110.0, 4.5, 1.5, 18.0],    # 向北方向，最长延迟
            [120.0, 3.8, 4.0, 10.0],    # 向西北偏西
            [85.0, 4.2, 2.5, 16.0],     # 向北偏西
            [95.0, 3.0, 3.5, 14.0],     # 向西南方向
            [105.0, 5.0, 1.0, 19.0],    # 向北方向
        ]
    else:  # FY3
        # FY3位置: [6000, -3000, 700] - 专门设计避免地面引爆的种子
        heuristic_seeds = [
            [140.0, 2.79, 55.0, 3.0],
            [130.0, 2.62, 58.0, 2.0],
            # 战术二：侧翼机动
            [140.0, 1.75, 50.0, 8.0],
            [130.0, 2.0, 48.0, 9.5],
            # 战术三：远程打击
            [140.0, 3.0, 8.0, 11.0],
            # 再加一个中庸的种子
            [120.0, 2.2, 45.0, 6.0]
        ]

    # 生成初始种群
    TOTAL_POPSIZE = 5000
    num_seeds = len(heuristic_seeds)
    num_random = TOTAL_POPSIZE - num_seeds

    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    # 为FY3生成更保守的随机种群
    sampler = qmc.LatinHypercube(d=len(bounds), seed=np.random.randint(1, 1000))
    random_init_unit_scale = sampler.random(n=num_random)

    if uav_name == "FY3":
        # 对FY3的随机种群进行特殊处理，偏向较短的延迟时间
        # 调整延迟时间分布，使其更多集中在较短范围
        delay_indices = 3  # t_explode_delay的索引
        random_init_unit_scale[:, delay_indices] = random_init_unit_scale[:, delay_indices] ** 2  # 平方操作使分布偏向较小值

    scaled_random_init = qmc.scale(random_init_unit_scale, lower_bounds, upper_bounds)

    initial_population = np.vstack([heuristic_seeds, scaled_random_init])

    print(f"初始种群大小: {TOTAL_POPSIZE} (包含{num_seeds}个启发式种子)")
    if uav_name == "FY3":
        print("  > FY3特殊处理: 缩短延迟时间边界和偏向短延迟的随机分布")

    # 测试几个种子的有效性
    print("测试启发式种子有效性:")
    valid_seeds = 0
    for i, seed in enumerate(heuristic_seeds):
        test_value = objective_function(seed)
        if test_value < -1e-6:
            valid_seeds += 1
        print(f"  种子{i+1}: {test_value:.6f}")

    print(f"有效种子数量: {valid_seeds}/{len(heuristic_seeds)}")

    print("开始优化...")

    start_time = time.time()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 差分进化优化
    result = differential_evolution(
        objective_function,
        bounds,
        init=initial_population,
        strategy='best1bin',
        maxiter=3000,
        tol=1e-6,
        recombination=0.7,
        mutation=(1.0, 1.9),
        disp=True,
        workers=-1,
        seed=42,
        polish=True,
        atol=1e-8
    )

    end_time = time.time()
    print(f"\n{uav_name}优化完成，耗时: {end_time - start_time:.2f} 秒")

    if result.success and result.fun < -1e-6:
        best_params = result.x
        max_duration = -result.fun

        print_solution_details(best_params, max_duration, uav_name, uav_initial_pos)

        print(f"\n{uav_name}最优参数:")
        seed_string = ", ".join([f"{val:.6f}" for val in best_params])
        print(f"[{seed_string}]")

        return best_params, max_duration

    else:
        print(f"\n❌ {uav_name}优化失败或未找到有效解")
        print(f"最佳找到值: {-result.fun:.6f} 秒")
        print(f"最佳参数: {result.x}")

        # 即使没有找到最优解，也返回最佳尝试
        if -result.fun > 0:
            return result.x, -result.fun
        return None, 0.0

if __name__ == "__main__":
    print("🎯 FY2和FY3无人机单独最优策略求解")
    print("=" * 60)
    print(f"导弹总飞行时间: {MISSILE_FLIGHT_TIME:.2f} s")
    print(f"时间步长: {OPTIMIZER_TIME_STEP:.3f} s")

    # 选择性运行配置
    SOLVE_FY2 = False   # 设为False跳过FY2优化
    SOLVE_FY3 = True   # 设为False跳过FY3优化

    print(f"\n运行配置: FY2={'启用' if SOLVE_FY2 else '跳过'}, FY3={'启用' if SOLVE_FY3 else '跳过'}")

    fy2_params, fy2_duration = None, 0.0
    fy3_params, fy3_duration = None, 0.0

    # 优化FY2
    if SOLVE_FY2:
        fy2_params, fy2_duration = optimize_single_uav("FY2", P_FY2_0)
    else:
        print("\n⏭️  跳过FY2优化")

    # 优化FY3
    if SOLVE_FY3:
        fy3_params, fy3_duration = optimize_single_uav("FY3", P_FY3_0)
    else:
        print("\n⏭️  跳过FY3优化")

    # 汇总结果
    print("\n" + "=" * 80)
    print("📊 单独策略汇总结果:")
    print("=" * 80)

    if SOLVE_FY2:
        if fy2_params is not None:
            print(f"FY2最优遮蔽时长: {fy2_duration:.6f} 秒")
            fy2_seed = ", ".join([f"{val:.6f}" for val in fy2_params])
            print(f"FY2最优参数: [{fy2_seed}]")
        else:
            print("FY2: 未找到有效解")
    else:
        print("FY2: 跳过优化")

    print()

    if SOLVE_FY3:
        if fy3_params is not None:
            print(f"FY3最优遮蔽时长: {fy3_duration:.6f} 秒")
            fy3_seed = ", ".join([f"{val:.6f}" for val in fy3_params])
            print(f"FY3最优参数: [{fy3_seed}]")
        else:
            print("FY3: 未找到有效解")
    else:
        print("FY3: 跳过优化")

    print("=" * 80)

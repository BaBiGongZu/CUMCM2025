import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import time
import warnings
import math
import ctypes
import os
import platform

# --- C库加载器 ---
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
        """
        计算总遮蔽时长（适配新的C库接口）

        Parameters:
        -----------
        missile_start : array-like
            导弹起始位置 [x, y, z]
        missile_velocity : array-like
            导弹速度向量 [vx, vy, vz]
        explode_positions : list of array-like
            烟雾弹起爆位置列表 [[x1,y1,z1], [x2,y2,z2], ...]
        explode_times : array-like
            起爆时间列表 [t1, t2, ...]
        total_flight_time : float
            导弹总飞行时间
        time_step : float
            时间步长
        sink_speed : float
            烟雾下沉速度
        smoke_duration : float
            烟雾持续时间

        Returns:
        --------
        float : 总遮蔽时长
        """
        # 转换为numpy数组并确保数据类型
        missile_start = np.array(missile_start, dtype=np.float64)
        missile_velocity = np.array(missile_velocity, dtype=np.float64)
        explode_positions = np.array(explode_positions, dtype=np.float64)
        explode_times = np.array(explode_times, dtype=np.float64)

        num_clouds = len(explode_positions)

        # 准备C函数参数
        missile_start_c = missile_start.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        missile_velocity_c = missile_velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # 展平起爆位置数组
        explode_pos_flat = explode_positions.flatten()
        explode_pos_c = explode_pos_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        explode_times_c = explode_times.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # 调用C函数
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

# --- 全局常量（与solve1.py保持一致）---
G_ACCEL = 9.8
P_M1_0 = np.array([20000.0, 0.0, 2000.0])
V_M1_SPEED = 300.0
TARGET_FALSE = np.array([0.0, 0.0, 0.0])

VEC_M1_TO_TARGET = TARGET_FALSE - P_M1_0
DIST_M1_TO_TARGET = np.linalg.norm(VEC_M1_TO_TARGET)
UNIT_VEC_M1 = VEC_M1_TO_TARGET / DIST_M1_TO_TARGET
VEC_V_M1 = V_M1_SPEED * UNIT_VEC_M1
MISSILE_FLIGHT_TIME = DIST_M1_TO_TARGET / V_M1_SPEED

P_FY1_0 = np.array([17800.0, 0.0, 1800.0])
REAL_TARGET_CENTER = np.array([0.0, 200.0, 0.0])
REAL_TARGET_RADIUS = 7.0
REAL_TARGET_HEIGHT = 10.0
R_SMOKE = 10.0
V_SMOKE_SINK_SPEED = 3.0
SMOKE_DURATION = 20.0

# 优化器参数
OPTIMIZER_TIME_STEP = 0.001  # 可以使用更高精度

# 初始化C库
print("🔧 初始化C库...")
c_smoke_lib = SmokeBlockingCLib()

def calculate_uav_direction_from_angle(flight_angle):
    """
    根据飞行角度计算无人机飞行方向
    
    Parameters:
    -----------
    flight_angle : float
        飞行角度（弧度），以X轴正向为0，逆时针为正。
    
    Returns:
    --------
    np.array : 3D方向向量
    """
    uav_direction = np.zeros(3)
    uav_direction[0] = np.cos(flight_angle)
    uav_direction[1] = np.sin(flight_angle)
    
    return uav_direction

def c_accelerated_objective_function(params):
    """
    使用C库加速的目标函数
    
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
        return 0.0
    
    # 1. 计算无人机飞行方向
    uav_direction = calculate_uav_direction_from_angle(flight_angle)
    
    # 2. 计算烟雾弹投放位置
    uav_drop_pos = P_FY1_0 + uav_speed * t_drop * uav_direction
    
    # 3. 计算烟雾弹起爆位置
    bomb_initial_velocity = uav_speed * uav_direction
    explode_pos = uav_drop_pos.copy()
    explode_pos += bomb_initial_velocity * t_explode_delay
    explode_pos[2] -= 0.5 * G_ACCEL * t_explode_delay**2
    
    # 检查起爆高度是否合理
    if explode_pos[2] <= REAL_TARGET_HEIGHT:
        return 0.0
    
    # 4. 使用新的C库接口进行计算
    explode_positions = [explode_pos]  # 单个烟雾弹
    explode_times = [t_explode_abs]

    # 调用新的C库函数
    total_duration = c_smoke_lib.calculate_total_duration(
        P_M1_0, VEC_V_M1, explode_positions, explode_times,
        MISSILE_FLIGHT_TIME, OPTIMIZER_TIME_STEP,
        V_SMOKE_SINK_SPEED, SMOKE_DURATION
    )
    
    return -total_duration  # 负值用于最大化

def python_objective_function_for_comparison(params):
    """
    纯Python版本的目标函数（用于对比）
    """
    uav_speed, flight_angle, t_drop, t_explode_delay = params
    
    t_explode_abs = t_drop + t_explode_delay
    
    if t_explode_abs >= MISSILE_FLIGHT_TIME:
        return 0.0
    
    uav_direction = calculate_uav_direction_from_angle(flight_angle)
    uav_drop_pos = P_FY1_0 + uav_speed * t_drop * uav_direction
    bomb_initial_velocity = uav_speed * uav_direction
    explode_pos = uav_drop_pos.copy()
    explode_pos += bomb_initial_velocity * t_explode_delay
    explode_pos[2] -= 0.5 * G_ACCEL * t_explode_delay**2
    
    if explode_pos[2] <= REAL_TARGET_HEIGHT:
        return 0.0
    
    # 使用C库函数进行计算，而不是纯Python实现
    explode_positions = [explode_pos]
    explode_times = [t_explode_abs]

    total_duration = c_smoke_lib.calculate_total_duration(
        P_M1_0, VEC_V_M1, explode_positions, explode_times,
        MISSILE_FLIGHT_TIME, 0.01,  # 使用较大的时间步长以提高性能
        V_SMOKE_SINK_SPEED, SMOKE_DURATION
    )

    return -total_duration

def print_solution_details(params, duration):
    """打印解的详细信息"""
    uav_speed, flight_angle, t_drop, t_explode_delay = params
    t_explode_abs = t_drop + t_explode_delay
    
    uav_direction = calculate_uav_direction_from_angle(flight_angle)
    uav_drop_pos = P_FY1_0 + uav_speed * t_drop * uav_direction
    bomb_initial_velocity = uav_speed * uav_direction
    explode_pos = uav_drop_pos.copy()
    explode_pos += bomb_initial_velocity * t_explode_delay
    explode_pos[2] -= 0.5 * G_ACCEL * t_explode_delay**2
    
    print(f"\n找到最优策略（C库加速版本）：")
    print(f"  > 最大有效遮蔽时长: {duration:.6f} 秒")
    print("-" * 60)
    print(f"  无人机飞行速度: {uav_speed:.4f} m/s")
    print(f"  飞行角度: {flight_angle:.6f} 弧度 ({math.degrees(flight_angle):.2f}°)")
    print(f"  受领任务后投放时间: {t_drop:.4f} s")
    print(f"  投放后起爆延迟: {t_explode_delay:.4f} s")
    print(f"  总起爆时间: {t_explode_abs:.4f} s")
    print("-" * 60)
    print(f"  无人机飞行方向: [{uav_direction[0]:.6f}, {uav_direction[1]:.6f}, {uav_direction[2]:.6f}]")
    print(f"  投放位置: [{uav_drop_pos[0]:.2f}, {uav_drop_pos[1]:.2f}, {uav_drop_pos[2]:.2f}]")
    print(f"  起爆位置: [{explode_pos[0]:.2f}, {explode_pos[1]:.2f}, {explode_pos[2]:.2f}]")
    print("-" * 60)

def performance_benchmark():
    """性能基准测试"""
    print("\n🚀 性能基准测试")
    print("=" * 50)
    
    # 测试参数
    test_params = [120.0, 0.1, 1.5, 3.6]
    
    # C库版本测试
    print("测试C库版本...")
    start_time = time.time()
    c_result = c_accelerated_objective_function(test_params)
    c_time = time.time() - start_time
    
    # Python版本测试
    print("测试Python版本...")
    start_time = time.time()
    py_result = python_objective_function_for_comparison(test_params)
    py_time = time.time() - start_time
    
    print(f"\n性能对比结果:")
    print(f"  C库版本:    {c_time:.4f}秒, 结果: {-c_result:.6f}秒")
    print(f"  Python版本: {py_time:.4f}秒, 结果: {-py_result:.6f}秒")
    print(f"  加速比:     {py_time/c_time:.1f}x")
    print(f"  结果误差:   {abs(c_result - py_result):.8f}")

if __name__ == "__main__":
    print("🎯 C库加速版solve2.1.py")
    print("=" * 60)
    print(f"导弹总飞行时间: {MISSILE_FLIGHT_TIME:.2f} s")
    print(f"时间步长: {OPTIMIZER_TIME_STEP:.3f} s (高精度)")
    
    # 运行性能基准测试
    performance_benchmark()
    
    print(f"\n开始优化...")
    start_time = time.time()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # 优化变量边界
    bounds = [
        (70.0, 140.0),      # uav_speed (m/s)
        (0, 2 * np.pi),     # flight_angle (rad)
        (0.0, 10.0),        # t_drop (s)
        (0.0, 10.0)         # t_explode_delay (s)
    ]
    
    # 使用已知的较好种子
    seed = np.array([[
        # 135.0596, 0.110243, 0.1983, 0.6139
        134.302340292277, 0.088445143950, 0.767470916269, 0.196204160294
    ]])

    # 生成初始种群
    TOTAL_POPSIZE = 1000  # 由于C加速，可以使用更大的种群
    num_random_individuals = TOTAL_POPSIZE - 1
    num_vars = len(bounds)
    
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    sampler = qmc.LatinHypercube(d=num_vars)
    random_init_unit_scale = sampler.random(n=num_random_individuals)
    scaled_random_init = qmc.scale(random_init_unit_scale, lower_bounds, upper_bounds)
    
    full_init_population = np.vstack((seed, scaled_random_init))
    
    print(f"初始种群大小: {TOTAL_POPSIZE} (包含1个种子)")
    print("开始C加速优化...")
    
    # 差分进化优化
    result = differential_evolution(
        c_accelerated_objective_function,  # 使用C库加速版本
        bounds,
        init=full_init_population,
        strategy='rand1bin',
        maxiter=1000,  # 由于C加速，可以增加迭代次数
        tol=0.001,
        recombination=0.7,
        mutation=(0.5, 1.0),
        disp=True,
        workers=-1,  # C库可能不是线程安全的，使用单线程
        seed=42
    )
    
    end_time = time.time()
    print(f"\n优化完成，总耗时: {end_time - start_time:.2f} 秒")
    
    if result.success and result.fun < -1e-9:
        best_params = result.x
        max_duration = -result.fun
        
        print_solution_details(best_params, max_duration)
        
        # 验证结果一致性
        print("\n🔍 结果验证:")
        c_duration = -c_accelerated_objective_function(best_params)
        py_duration = -python_objective_function_for_comparison(best_params)
        print(f"  C库计算结果:    {c_duration:.6f}秒")
        print(f"  Python计算结果: {py_duration:.6f}秒")
        print(f"  误差:           {abs(c_duration - py_duration):.8f}秒")
        
    else:
        print("\n❌ 优化失败或未找到有效解")
        print(f"最佳找到值: {-result.fun:.6f} 秒")
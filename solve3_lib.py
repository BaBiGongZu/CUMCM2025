"""
问题三：利用无人机FY1投放3枚烟幕干扰弹实施对M1的干扰
使用C库加速计算的优化程序
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc
import time
import warnings
import math
import ctypes
import os
import platform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- C库加载器 ---
class MultiCloudSmokeBlockingLib:
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
        # 多烟雾弹检查函数
        self.c_lib.check_multiple_clouds_blocking.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # missile_pos
            ctypes.POINTER(ctypes.c_double),  # cloud_positions (作为一维数组传递)
            ctypes.c_int                      # num_clouds
        ]
        self.c_lib.check_multiple_clouds_blocking.restype = ctypes.c_bool
        
        # 多烟雾弹批量计算函数
        self.c_lib.calculate_multiple_clouds_blocking_duration.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # missile_start
            ctypes.POINTER(ctypes.c_double),  # missile_velocity
            ctypes.POINTER(ctypes.c_double),  # explode_positions (作为一维数组)
            ctypes.c_int,                     # num_clouds
            ctypes.c_double,                  # t_start
            ctypes.c_double,                  # t_end
            ctypes.c_double,                  # time_step
            ctypes.c_double                   # sink_speed
        ]
        self.c_lib.calculate_multiple_clouds_blocking_duration.restype = ctypes.c_double
        
        # 辅助函数
        self.c_lib.get_version.restype = ctypes.c_char_p
        self.c_lib.get_key_points_count.restype = ctypes.c_int
        
        self.c_lib.get_sampling_info.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int)
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
        print(f"关键点总数: {key_points_count}")
        print(f"采样分布: 底面{bottom.value} + 顶面{top.value} + 侧面{side.value} = {total.value}")
    
    def get_version(self):
        """获取库版本"""
        return self.c_lib.get_version().decode('utf-8')
    
    def get_key_points_count(self):
        """获取关键点数量"""
        return self.c_lib.get_key_points_count()
    
    def check_multiple_clouds_blocking(self, missile_pos, cloud_positions):
        """
        检查多烟雾弹遮蔽
        
        Parameters:
        -----------
        missile_pos : array-like
            导弹位置 [x, y, z]
        cloud_positions : array-like
            烟雾弹位置列表 [[x1,y1,z1], [x2,y2,z2], ...]
            
        Returns:
        --------
        bool : 是否完全遮蔽
        """
        cloud_positions = np.array(cloud_positions)
        num_clouds = len(cloud_positions)
        
        missile_pos_c = (ctypes.c_double * 3)(*missile_pos)
        
        # 将二维数组展平为一维数组传递给C
        cloud_flat = cloud_positions.flatten()
        cloud_positions_c = (ctypes.c_double * len(cloud_flat))(*cloud_flat)
        
        return self.c_lib.check_multiple_clouds_blocking(
            missile_pos_c, cloud_positions_c, num_clouds
        )
    
    def calculate_multiple_clouds_duration(self, missile_start, missile_velocity, 
                                         explode_positions, t_start, t_end, 
                                         time_step, sink_speed):
        """计算多烟雾弹遮蔽时长"""
        explode_positions = np.array(explode_positions)
        num_clouds = len(explode_positions)
        
        missile_start_c = (ctypes.c_double * 3)(*missile_start)
        missile_velocity_c = (ctypes.c_double * 3)(*missile_velocity)
        
        # 将二维数组展平
        explode_flat = explode_positions.flatten()
        explode_positions_c = (ctypes.c_double * len(explode_flat))(*explode_flat)
        
        return self.c_lib.calculate_multiple_clouds_blocking_duration(
            missile_start_c,
            missile_velocity_c,
            explode_positions_c,
            num_clouds,
            ctypes.c_double(t_start),
            ctypes.c_double(t_end),
            ctypes.c_double(time_step),
            ctypes.c_double(sink_speed)
        )


# --- 全局常量 ---
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
OPTIMIZER_TIME_STEP = 0.01  # 适当调整以平衡精度和性能
NUM_SMOKE_BOMBS = 3  # 问题三要求投放3枚烟幕干扰弹

# C库全局变量（延迟初始化）
c_smoke_lib = None

def get_smoke_lib():
    """获取C库实例，使用单例模式避免重复初始化"""
    global c_smoke_lib
    if c_smoke_lib is None:
        print("🔧 初始化多烟雾弹C库...")
        c_smoke_lib = MultiCloudSmokeBlockingLib()
    return c_smoke_lib

# 为多进程优化：将C库调用封装为独立函数
def calculate_multiple_clouds_duration_mp(missile_start, missile_velocity, explode_positions, 
                                         t_start, t_end, time_step, sink_speed):
    """
    多进程安全的烟雾弹遮蔽时长计算函数
    每个进程会独立初始化C库
    """
    lib = get_smoke_lib()
    return lib.calculate_multiple_clouds_duration(
        missile_start, missile_velocity, explode_positions, 
        t_start, t_end, time_step, sink_speed
    )

def init_worker_process():
    """
    工作进程初始化函数
    在每个工作进程启动时调用，确保C库正确初始化
    """
    global c_smoke_lib
    c_smoke_lib = None  # 重置全局变量
    # 预先初始化C库以避免竞态条件
    try:
        get_smoke_lib()
        print(f"✅ 工作进程 {os.getpid()} C库初始化成功")
    except Exception as e:
        print(f"❌ 工作进程 {os.getpid()} C库初始化失败: {e}")

def three_bombs_objective_function_mp(params):
    """
    多进程优化版本的目标函数
    使用独立的C库调用函数，避免序列化问题
    """
    try:
        # 解码参数
        trajectory_info = decode_params_to_trajectories(params)
        explode_times = trajectory_info['explode_times']
        trajectories = trajectory_info['trajectories']
        
        # 检查所有起爆时间是否合理
        for i, t_explode in enumerate(explode_times):
            if t_explode >= MISSILE_FLIGHT_TIME:
                return 0.0  # 起爆时间过晚
            
            if trajectories[i]['explode_pos'][2] <= REAL_TARGET_HEIGHT:
                return 0.0  # 起爆高度过低
        
        # 计算所有烟幕弹的有效时间段
        smoke_periods = []
        for i, t_explode in enumerate(explode_times):
            t_start = t_explode
            t_end = min(t_explode + SMOKE_DURATION, MISSILE_FLIGHT_TIME)
            if t_start < t_end:
                smoke_periods.append((t_start, t_end, i))
        
        if not smoke_periods:
            return 0.0
        
        # 创建时间事件列表
        events = []
        for t_start, t_end, bomb_idx in smoke_periods:
            events.append((t_start, 'start', bomb_idx))
            events.append((t_end, 'end', bomb_idx))
        
        # 按时间排序
        events.sort()
        
        total_duration = 0.0
        active_bombs = set()
        last_time = None
        
        # 扫描时间线
        for event_time, event_type, bomb_idx in events:
            # 如果有活跃的烟幕弹，计算这段时间的遮蔽效果
            if active_bombs and last_time is not None and event_time > last_time:
                active_positions = [trajectories[i]['explode_pos'] for i in active_bombs]
                
                # 使用多进程安全的C库调用
                duration = calculate_multiple_clouds_duration_mp(
                    P_M1_0, VEC_V_M1, active_positions,
                    last_time, event_time, OPTIMIZER_TIME_STEP, V_SMOKE_SINK_SPEED
                )
                total_duration += duration
            
            # 更新活跃烟幕弹集合
            if event_type == 'start':
                active_bombs.add(bomb_idx)
            else:
                active_bombs.discard(bomb_idx)
            
            last_time = event_time
        
        return -total_duration  # 负值用于最大化
        
    except Exception as e:
        print(f"目标函数计算出错: {e}")
        return 0.0

def calculate_uav_direction_from_angle(flight_angle):
    """根据飞行角度计算无人机飞行方向"""
    uav_direction = np.zeros(3)
    uav_direction[0] = np.cos(flight_angle)
    uav_direction[1] = np.sin(flight_angle)
    return uav_direction

def decode_params_to_trajectories(params):
    """
    将优化参数解码为3枚烟幕干扰弹的轨迹信息
    
    Parameters:
    -----------
    params : array-like
        优化参数 [uav_speed, flight_angle, t_drop1, t_delay1, interval2, t_delay2, interval3, t_delay3]
        - uav_speed: 无人机飞行速度
        - flight_angle: 无人机飞行角度
        - t_drop1: 第1枚烟幕弹的投放时间(绝对时间)
        - t_delay1: 第1枚烟幕弹的起爆延迟
        - interval2: 第2枚相对第1枚的投放间隔时间
        - t_delay2: 第2枚烟幕弹的起爆延迟
        - interval3: 第3枚相对第2枚的投放间隔时间
        - t_delay3: 第3枚烟幕弹的起爆延迟
    
    Returns:
    --------
    dict: 包含轨迹信息的字典
    """
    uav_speed, flight_angle, t_drop1, t_delay1, interval2, t_delay2, interval3, t_delay3 = params
    
    # 计算绝对投放时间
    t_drop2 = t_drop1 + interval2
    t_drop3 = t_drop2 + interval3
    
    # 计算无人机飞行方向
    uav_direction = calculate_uav_direction_from_angle(flight_angle)
    
    trajectories = []
    explode_times = []
    
    for i, (t_drop, t_delay) in enumerate([(t_drop1, t_delay1), (t_drop2, t_delay2), (t_drop3, t_delay3)]):
        # 投放位置
        drop_pos = P_FY1_0 + uav_speed * t_drop * uav_direction
        
        # 起爆位置
        bomb_initial_velocity = uav_speed * uav_direction
        explode_pos = drop_pos.copy()
        explode_pos += bomb_initial_velocity * t_delay
        explode_pos[2] -= 0.5 * G_ACCEL * t_delay**2
        
        # 起爆时间
        t_explode = t_drop + t_delay
        
        trajectories.append({
            'drop_pos': drop_pos,
            'explode_pos': explode_pos,
            't_drop': t_drop,
            't_explode': t_explode,
            't_delay': t_delay
        })
        
        explode_times.append(t_explode)
    
    return {
        'uav_speed': uav_speed,
        'flight_angle': flight_angle,
        'uav_direction': uav_direction,
        'trajectories': trajectories,
        'explode_times': explode_times,
        'drop_intervals': [interval2, interval3]  # 添加间隔信息用于调试
    }

def three_bombs_objective_function(params):
    """
    三枚烟幕干扰弹的C库加速目标函数
    
    Parameters:
    -----------
    params : array-like
        [uav_speed, flight_angle, t_drop1, t_delay1, t_drop2, t_delay2, t_drop3, t_delay3]
    """
    try:
        # 解码参数
        trajectory_info = decode_params_to_trajectories(params)
        explode_times = trajectory_info['explode_times']
        trajectories = trajectory_info['trajectories']
        
        # 检查所有起爆时间是否合理
        for i, t_explode in enumerate(explode_times):
            if t_explode >= MISSILE_FLIGHT_TIME:
                return 0.0  # 起爆时间过晚
            
            if trajectories[i]['explode_pos'][2] <= REAL_TARGET_HEIGHT:
                return 0.0  # 起爆高度过低
        
        # 计算所有烟幕弹的有效时间段
        smoke_periods = []
        for i, t_explode in enumerate(explode_times):
            t_start = t_explode
            t_end = min(t_explode + SMOKE_DURATION, MISSILE_FLIGHT_TIME)
            if t_start < t_end:
                smoke_periods.append((t_start, t_end, i))
        
        if not smoke_periods:
            return 0.0
        
        # 创建时间事件列表
        events = []
        for t_start, t_end, bomb_idx in smoke_periods:
            events.append((t_start, 'start', bomb_idx))
            events.append((t_end, 'end', bomb_idx))
        
        # 按时间排序
        events.sort()
        
        total_duration = 0.0
        active_bombs = set()
        last_time = None
        
        # 扫描时间线
        for event_time, event_type, bomb_idx in events:
            # 如果有活跃的烟幕弹，计算这段时间的遮蔽效果
            if active_bombs and last_time is not None and event_time > last_time:
                active_positions = [trajectories[i]['explode_pos'] for i in active_bombs]
                
                # 使用C库计算这个时间段的遮蔽时长
                duration = get_smoke_lib().calculate_multiple_clouds_duration(
                    P_M1_0, VEC_V_M1, active_positions,
                    last_time, event_time, OPTIMIZER_TIME_STEP, V_SMOKE_SINK_SPEED
                )
                total_duration += duration
            
            # 更新活跃烟幕弹集合
            if event_type == 'start':
                active_bombs.add(bomb_idx)
            else:
                active_bombs.discard(bomb_idx)
            
            last_time = event_time
        
        return -total_duration  # 负值用于最大化
        
    except Exception as e:
        print(f"目标函数计算出错: {e}")
        return 0.0

def validate_trajectory_constraints(params, verbose=False):
    """
    验证轨迹约束条件
    
    Returns:
    --------
    bool: 是否满足约束
    """
    uav_speed, flight_angle, t_drop1, t_delay1, interval2, t_delay2, interval3, t_delay3 = params
    
    constraint_violations = []
    
    # 检查投放时间间隔（现在由参数结构自然保证 >= 0）
    if interval2 < 1.0:
        constraint_violations.append(f"第2枚投放间隔不足: {interval2:.3f}s < 1.0s")
    
    if interval3 < 1.0:
        constraint_violations.append(f"第3枚投放间隔不足: {interval3:.3f}s < 1.0s")
    
    # 检查速度范围
    if uav_speed < 70 or uav_speed > 140:
        constraint_violations.append(f"无人机速度超出范围: {uav_speed:.3f} m/s (要求: 70-140 m/s)")
    
    # 检查时间参数的基本合理性
    time_params = [(t_drop1, "第1枚投放时间"), (t_delay1, "第1枚延迟时间"), 
                   (interval2, "第2枚投放间隔"), (t_delay2, "第2枚延迟时间"),
                   (interval3, "第3枚投放间隔"), (t_delay3, "第3枚延迟时间")]
    
    for t, name in time_params:
        if t < 0.1:
            constraint_violations.append(f"{name}过小: {t:.3f}s < 0.1s")
    
    # 检查起爆时间和高度
    trajectory_info = decode_params_to_trajectories(params)
    for i, traj in enumerate(trajectory_info['trajectories'], 1):
        if traj['t_explode'] >= MISSILE_FLIGHT_TIME:
            constraint_violations.append(
                f"第{i}枚起爆时间过晚: {traj['t_explode']:.3f}s >= {MISSILE_FLIGHT_TIME:.3f}s"
            )
        
        if traj['explode_pos'][2] <= REAL_TARGET_HEIGHT:
            constraint_violations.append(
                f"第{i}枚起爆高度过低: {traj['explode_pos'][2]:.3f}m <= {REAL_TARGET_HEIGHT:.3f}m"
            )
    
    if verbose and constraint_violations:
        print(f"🚫 约束违反 ({len(constraint_violations)}项):")
        for violation in constraint_violations:
            print(f"    {violation}")
    
    return len(constraint_violations) == 0

def debug_time_calculation(params, detailed=True):
    """
    调试时间计算过程
    """
    trajectory_info = decode_params_to_trajectories(params)
    explode_times = trajectory_info['explode_times']
    trajectories = trajectory_info['trajectories']
    
    print("\n🔍 详细时间计算调试:")
    
    # 计算所有烟幕弹的有效时间段
    smoke_periods = []
    for i, t_explode in enumerate(explode_times):
        t_start = t_explode
        t_end = min(t_explode + SMOKE_DURATION, MISSILE_FLIGHT_TIME)
        if t_start < t_end:
            smoke_periods.append((t_start, t_end, i))
            print(f"  烟幕弹{i+1}: {t_start:.3f}s - {t_end:.3f}s (持续{t_end-t_start:.3f}s)")
    
    # 创建时间事件列表
    events = []
    for t_start, t_end, bomb_idx in smoke_periods:
        events.append((t_start, 'start', bomb_idx))
        events.append((t_end, 'end', bomb_idx))
    
    # 按时间排序
    events.sort()
    
    print(f"\n时间事件序列:")
    for event_time, event_type, bomb_idx in events:
        print(f"  {event_time:.3f}s: 烟幕弹{bomb_idx+1} {event_type}")
    
    total_duration = 0.0
    active_bombs = set()
    last_time = None
    segment_count = 0
    
    print(f"\n时间段遮蔽计算:")
    
    # 扫描时间线
    for event_time, event_type, bomb_idx in events:
        # 如果有活跃的烟幕弹，计算这段时间的遮蔽效果
        if active_bombs and last_time is not None and event_time > last_time:
            segment_count += 1
            active_positions = [trajectories[i]['explode_pos'] for i in active_bombs]
            
            # 使用C库计算这个时间段的遮蔽时长
            duration = get_smoke_lib().calculate_multiple_clouds_duration(
                P_M1_0, VEC_V_M1, active_positions,
                last_time, event_time, OPTIMIZER_TIME_STEP, V_SMOKE_SINK_SPEED
            )
            
            total_duration += duration
            active_bomb_list = sorted(list(active_bombs))
            
            print(f"  段{segment_count}: {last_time:.3f}s - {event_time:.3f}s, 活跃烟幕弹{[i+1 for i in active_bomb_list]}, 遮蔽时长: {duration:.6f}s")
            
            if detailed and duration > 0:
                # 详细分析这个时间段
                time_span = event_time - last_time
                coverage_ratio = duration / time_span if time_span > 0 else 0
                print(f"    总时间跨度: {time_span:.3f}s, 遮蔽覆盖率: {coverage_ratio:.2%}")
        
        # 更新活跃烟幕弹集合
        if event_type == 'start':
            active_bombs.add(bomb_idx)
            print(f"    -> 烟幕弹{bomb_idx+1}开始生效")
        else:
            active_bombs.discard(bomb_idx)
            print(f"    -> 烟幕弹{bomb_idx+1}结束生效")
        
        last_time = event_time
    
    print(f"\n总遮蔽时长: {total_duration:.6f} 秒")
    return total_duration

def print_solution_details(params, duration):
    """打印解的详细信息 (新版本：相对时间间隔)"""
    trajectory_info = decode_params_to_trajectories(params)
    uav_speed, flight_angle, t_drop1, t_delay1, interval2, t_delay2, interval3, t_delay3 = params
    
    print(f"\n找到最优三烟幕弹策略（C库加速版本 - 相对时间间隔）：")
    print(f"  > 最大有效遮蔽时长: {duration:.6f} 秒")
    print("=" * 80)
    
    print(f"无人机参数:")
    print(f"  飞行速度: {trajectory_info['uav_speed']:.4f} m/s")
    print(f"  飞行角度: {trajectory_info['flight_angle']:.6f} 弧度 ({math.degrees(trajectory_info['flight_angle']):.2f}°)")
    print(f"  飞行方向: [{trajectory_info['uav_direction'][0]:.6f}, {trajectory_info['uav_direction'][1]:.6f}, {trajectory_info['uav_direction'][2]:.6f}]")
    
    print(f"\n投放时间参数 (新结构):")
    print(f"  第1枚投放时间: {t_drop1:.4f} s (绝对时间)")
    print(f"  第2枚投放间隔: {interval2:.4f} s (相对第1枚)")
    print(f"  第3枚投放间隔: {interval3:.4f} s (相对第2枚)")
    
    print(f"\n烟幕干扰弹详情:")
    for i, traj in enumerate(trajectory_info['trajectories'], 1):
        print(f"  第{i}枚烟幕弹:")
        print(f"    投放时间: {traj['t_drop']:.4f} s (绝对时间)")
        print(f"    起爆延迟: {traj['t_delay']:.4f} s")
        print(f"    起爆时间: {traj['t_explode']:.4f} s")
        print(f"    投放位置: [{traj['drop_pos'][0]:.2f}, {traj['drop_pos'][1]:.2f}, {traj['drop_pos'][2]:.2f}]")
        print(f"    起爆位置: [{traj['explode_pos'][0]:.2f}, {traj['explode_pos'][1]:.2f}, {traj['explode_pos'][2]:.2f}]")
    
    # 检查投放间隔
    drop_times = [traj['t_drop'] for traj in trajectory_info['trajectories']]
    drop_times.sort()
    print(f"\n投放时间序列: {[f'{t:.3f}s' for t in drop_times]}")
    actual_intervals = [drop_times[i] - drop_times[i-1] for i in range(1, len(drop_times))]
    print(f"实际投放间隔: {[f'{interval:.3f}s' for interval in actual_intervals]} (要求≥1.0s)")
    print(f"设定投放间隔: [{interval2:.3f}s, {interval3:.3f}s]")
    
    print("=" * 80)

def visualize_three_bombs_strategy(params):
    """可视化三烟幕弹策略"""
    trajectory_info = decode_params_to_trajectories(params)
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 绘制假目标和真目标
    ax.scatter(*TARGET_FALSE, color='blue', s=100, marker='x', label='Fake Target', alpha=0.8)
    
    # 绘制真目标圆柱体（简化为几个关键点）
    theta = np.linspace(0, 2*np.pi, 20)
    
    # 底面圆
    bottom_x = REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * np.cos(theta)
    bottom_y = REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * np.sin(theta)
    bottom_z = np.zeros_like(theta)
    ax.plot(bottom_x, bottom_y, bottom_z, 'g-', alpha=0.6, linewidth=2, label='Target Bottom')
    
    # 顶面圆
    top_x = bottom_x
    top_y = bottom_y
    top_z = np.full_like(theta, REAL_TARGET_HEIGHT)
    ax.plot(top_x, top_y, top_z, 'g-', alpha=0.6, linewidth=2, label='Target Top')
    
    # 绘制导弹轨迹
    missile_end_pos = P_M1_0 + VEC_V_M1 * MISSILE_FLIGHT_TIME
    ax.plot([P_M1_0[0], missile_end_pos[0]], 
            [P_M1_0[1], missile_end_pos[1]], 
            [P_M1_0[2], missile_end_pos[2]], 'r-', linewidth=3, label='Missile Trajectory', alpha=0.8)
    ax.scatter(*P_M1_0, color='red', s=150, marker='>', label='Missile Start')
    
    # 绘制无人机轨迹
    colors = ['cyan', 'magenta', 'orange']
    for i, traj in enumerate(trajectory_info['trajectories']):
        # 无人机到投放点
        ax.plot([P_FY1_0[0], traj['drop_pos'][0]], 
                [P_FY1_0[1], traj['drop_pos'][1]], 
                [P_FY1_0[2], traj['drop_pos'][2]], 
                color=colors[i], linestyle='--', alpha=0.7, linewidth=2,
                label=f'UAV Path to Drop {i+1}')
        
        # 投放点
        ax.scatter(*traj['drop_pos'], color=colors[i], s=100, marker='v', 
                  label=f'Drop Point {i+1}', alpha=0.8)
        
        # 烟幕弹轨迹
        ax.plot([traj['drop_pos'][0], traj['explode_pos'][0]], 
                [traj['drop_pos'][1], traj['explode_pos'][1]], 
                [traj['drop_pos'][2], traj['explode_pos'][2]], 
                color=colors[i], linestyle=':', linewidth=2, alpha=0.8,
                label=f'Bomb {i+1} Trajectory')
        
        # 起爆点和烟雾云
        explode_pos = traj['explode_pos']
        ax.scatter(*explode_pos, color=colors[i], s=200, marker='*', 
                  label=f'Explosion {i+1}', alpha=0.9)
        
        # 绘制烟雾云（简化为球体）
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        x = explode_pos[0] + R_SMOKE * np.outer(np.cos(u), np.sin(v))
        y = explode_pos[1] + R_SMOKE * np.outer(np.sin(u), np.sin(v))
        z = explode_pos[2] + R_SMOKE * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color=colors[i], alpha=0.2)
    
    # 起始点
    ax.scatter(*P_FY1_0, color='black', s=150, marker='s', label='UAV Start', alpha=0.9)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Three Smoke Bombs Interference Strategy\n(C-Library Accelerated Optimization)')
    
    # 图例
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🎯 问题三：三枚烟幕干扰弹C库加速优化程序")
    print("=" * 80)
    print(f"导弹总飞行时间: {MISSILE_FLIGHT_TIME:.2f} s")
    print(f"烟幕弹数量: {NUM_SMOKE_BOMBS}")
    print(f"时间步长: {OPTIMIZER_TIME_STEP:.3f} s")
    
    # 参数边界 - 新版本：使用相对时间间隔
    bounds = [
        (70.0, 140.0),     # uav_speed
        (0, 2 * np.pi),    # flight_angle
        (0.1, 15.0),       # t_drop1 (第1枚投放时间，绝对时间)
        (0.1, 20.0),       # t_delay1 (第1枚起爆延迟)
        (1.0, 10.0),       # interval2 (第2枚相对第1枚的投放间隔，≥1秒)
        (0.1, 20.0),       # t_delay2 (第2枚起爆延迟)
        (1.0, 10.0),       # interval3 (第3枚相对第2枚的投放间隔，≥1秒)
        (0.1, 20.0),       # t_delay3 (第3枚起爆延迟)
    ]
    
    print(f"参数边界说明 (新版本: 相对时间间隔):")
    print(f"  无人机速度: {bounds[0][0]:.1f} - {bounds[0][1]:.1f} m/s")
    print(f"  飞行角度: {bounds[1][0]:.3f} - {bounds[1][1]:.3f} rad")
    print(f"  第1枚投放时间: {bounds[2][0]:.1f} - {bounds[2][1]:.1f} s (绝对时间)")
    print(f"  第1枚起爆延迟: {bounds[3][0]:.1f} - {bounds[3][1]:.1f} s")
    print(f"  第2枚投放间隔: {bounds[4][0]:.1f} - {bounds[4][1]:.1f} s (相对第1枚)")
    print(f"  第2枚起爆延迟: {bounds[5][0]:.1f} - {bounds[5][1]:.1f} s")
    print(f"  第3枚投放间隔: {bounds[6][0]:.1f} - {bounds[6][1]:.1f} s (相对第2枚)")
    print(f"  第3枚起爆延迟: {bounds[7][0]:.1f} - {bounds[7][1]:.1f} s")
    print()
    
    # 初始化种群（使用一些启发式种子）
    TOTAL_POPSIZE = 1000
    
    # 创建一些启发式种子 - 新版本：使用相对时间间隔
    heuristic_seeds = [
        # 均匀时间分布策略 
        [120.0, 0.0, 1.0, 3.0, 2.0, 3.5, 2.0, 4.0],    # t_drop1=1.0, t_drop2=3.0, t_drop3=5.0
        [110.0, 0.1, 1.5, 2.5, 2.5, 3.0, 2.0, 3.5],    # t_drop1=1.5, t_drop2=4.0, t_drop3=6.0
        [130.0, -0.1, 0.5, 4.0, 1.5, 2.8, 1.5, 4.2],   # t_drop1=0.5, t_drop2=2.0, t_drop3=3.5
        # 快速连发策略
        [125.0, 0.05, 1.0, 2.0, 1.0, 2.5, 1.0, 3.0],   # t_drop1=1.0, t_drop2=2.0, t_drop3=3.0
        # 延迟爆炸策略
        [115.0, 0.0, 2.0, 5.0, 2.0, 5.5, 2.0, 6.0],    # t_drop1=2.0, t_drop2=4.0, t_drop3=6.0
        # 紧密间隔策略
        [105.0, 0.2, 0.5, 1.5, 1.0, 1.8, 1.0, 2.0],    # t_drop1=0.5, t_drop2=1.5, t_drop3=2.5
        # 分散时间策略
        [135.0, -0.2, 2.5, 3.0, 3.0, 4.0, 3.0, 3.5],   # t_drop1=2.5, t_drop2=5.5, t_drop3=8.5
        # 手动跑的优化
        [140, 0.078378, 0.102, 0.802, 1.160, 4.2, 1.18, 11]
    ]
    
    num_seeds = len(heuristic_seeds)
    num_random = TOTAL_POPSIZE - num_seeds
    
    # 生成随机个体
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    sampler = qmc.LatinHypercube(d=len(bounds))
    random_init_unit_scale = sampler.random(n=num_random)
    scaled_random_init = qmc.scale(random_init_unit_scale, lower_bounds, upper_bounds)
    
    # 合并种子和随机个体
    initial_population = np.vstack([heuristic_seeds, scaled_random_init])
    
    print(f"\n初始种群大小: {TOTAL_POPSIZE} (包含{num_seeds}个启发式种子)")
    
    # 选择执行模式
    USE_MULTIPROCESSING = True  # 设置为True启用多进程，False使用单进程
    
    if USE_MULTIPROCESSING:
        print("🚀 启用多进程加速优化...")
        import multiprocessing as mp
        num_workers = mp.cpu_count()  # 限制最大进程数
        print(f"  使用 {num_workers} 个工作进程")
        
        # 确保主进程C库已初始化
        get_smoke_lib()
        
        objective_func = three_bombs_objective_function_mp
        workers_setting = num_workers
    else:
        print("🔧 使用单进程优化...")
        # 确保C库已初始化
        get_smoke_lib()
        
        objective_func = three_bombs_objective_function
        workers_setting = 1
    
    start_time = time.time()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # 差分进化优化
    if USE_MULTIPROCESSING:
        # 多进程版本：让每个工作进程自己初始化C库
        result = differential_evolution(
            objective_func,
            bounds,
            init=initial_population,
            strategy='best1bin',
            maxiter=2000,
            tol=0.001,
            recombination=0.7,
            mutation=(0.5, 1.0),
            disp=True,
            workers=workers_setting,
            seed=42,
            atol=1e-6
        )
    else:
        # 单进程版本
        result = differential_evolution(
            objective_func,
            bounds,
            init=initial_population,
            strategy='rand1bin',
            maxiter=1000,
            tol=0.001,
            recombination=0.7,
            mutation=(0.5, 1.0),
            disp=True,
            workers=workers_setting,
            seed=42,
            atol=1e-6
        )
    
    end_time = time.time()
    print(f"\n优化完成，总耗时: {end_time - start_time:.2f} 秒")
    
    if result.success and result.fun < -1e-6:
        best_params = result.x
        max_duration = -result.fun
        
        print(f"\n🎯 优化收敛成功!")
        print(f"  函数值: {result.fun:.8f}")
        print(f"  最大遮蔽时长: {max_duration:.6f} 秒")
        print(f"  迭代次数: {result.nit}")
        print(f"  函数评估次数: {result.nfev}")
        
        # 详细验证约束条件
        print(f"\n🔍 详细约束验证:")
        is_valid = validate_trajectory_constraints(best_params, verbose=True)
        
        if is_valid:
            print(f"✅ 所有约束条件满足!")
            print_solution_details(best_params, max_duration)
            
            # 可视化结果
            #print("\n生成策略可视化...")
            #visualize_three_bombs_strategy(best_params)
            
            # 额外分析
            trajectory_info = decode_params_to_trajectories(best_params)
            explode_times = trajectory_info['explode_times']
            print(f"\n📊 策略分析:")
            print(f"  起爆时间跨度: {max(explode_times) - min(explode_times):.3f} 秒")
            print(f"  最早起爆: {min(explode_times):.3f} 秒")
            print(f"  最晚起爆: {max(explode_times):.3f} 秒")
            
            # 计算单独每枚烟幕弹的贡献
            print(f"\n🔍 单枚烟幕弹分析:")
            total_global_start = min(explode_times)
            total_global_end = min(max(explode_times) + SMOKE_DURATION, MISSILE_FLIGHT_TIME)
            
            for i, traj in enumerate(trajectory_info['trajectories'], 1):
                # 计算这枚烟幕弹的有效时间段
                bomb_start = traj['t_explode']
                bomb_end = min(traj['t_explode'] + SMOKE_DURATION, MISSILE_FLIGHT_TIME)
                
                if bomb_start < bomb_end:
                    single_duration = get_smoke_lib().calculate_multiple_clouds_duration(
                        P_M1_0, VEC_V_M1, [traj['explode_pos']],
                        bomb_start, bomb_end,
                        OPTIMIZER_TIME_STEP, V_SMOKE_SINK_SPEED
                    )
                    print(f"    第{i}枚单独遮蔽时长: {single_duration:.6f} 秒 (有效期: {bomb_start:.3f}s - {bomb_end:.3f}s)")
                else:
                    print(f"    第{i}枚单独遮蔽时长: 0.000000 秒 (无效)")
            
            # 验证总时长计算 - 使用原始目标函数确保一致性
            print(f"\n🔍 总时长验证:")
            print(f"  全局有效时间段: {total_global_start:.3f}s - {total_global_end:.3f}s")
            
            # 重新计算总时长用于验证（使用单进程版本确保一致性）
            verification_duration = -three_bombs_objective_function(best_params)
            print(f"  优化结果时长: {max_duration:.6f} 秒")
            print(f"  验证计算时长: {verification_duration:.6f} 秒")
            print(f"  差异: {abs(max_duration - verification_duration):.8f} 秒")
            
            # 分析时间重叠情况
            print(f"\n🔍 时间重叠分析:")
            explode_times_sorted = sorted([(t, i) for i, t in enumerate(explode_times)])
            for i, (t_explode, bomb_idx) in enumerate(explode_times_sorted):
                t_end = min(t_explode + SMOKE_DURATION, MISSILE_FLIGHT_TIME)
                overlaps = []
                for j, (other_explode, other_idx) in enumerate(explode_times_sorted):
                    if i != j:
                        other_end = min(other_explode + SMOKE_DURATION, MISSILE_FLIGHT_TIME)
                        # 检查时间段重叠
                        overlap_start = max(t_explode, other_explode)
                        overlap_end = min(t_end, other_end)
                        if overlap_start < overlap_end:
                            overlaps.append((other_idx + 1, overlap_start, overlap_end))
                
                print(f"    第{bomb_idx + 1}枚 ({t_explode:.3f}s-{t_end:.3f}s):", end="")
                if overlaps:
                    for other_bomb, start, end in overlaps:
                        print(f" 与第{other_bomb}枚重叠({start:.3f}s-{end:.3f}s)", end="")
                else:
                    print(" 无重叠", end="")
                print()
            # 详细调试时间计算
            print(f"\n" + "="*50 + " 详细调试 " + "="*50)
            debug_duration = debug_time_calculation(best_params, detailed=True)
            print(f"调试计算总时长: {debug_duration:.6f} 秒")
            print("="*110)
            
        else:
            print("❌ 最优解不满足约束条件")
            print("\n详细参数信息:")
            uav_speed, flight_angle, t_drop1, t_delay1, t_drop2, t_delay2, t_drop3, t_delay3 = best_params
            print(f"  无人机速度: {uav_speed:.4f} m/s")
            print(f"  飞行角度: {flight_angle:.6f} rad ({math.degrees(flight_angle):.2f}°)")
            print(f"  投放时间: [{t_drop1:.3f}, {t_drop2:.3f}, {t_drop3:.3f}] s")
            print(f"  延迟时间: [{t_delay1:.3f}, {t_delay2:.3f}, {t_delay3:.3f}] s")
            
            # 计算投放间隔
            drop_times = [t_drop1, t_drop2, t_drop3]
            drop_times.sort()
            intervals = [drop_times[i] - drop_times[i-1] for i in range(1, len(drop_times))]
            print(f"  投放间隔: {[f'{interval:.3f}s' for interval in intervals]}")
            
            # 分析每枚烟幕弹的问题
            trajectory_info = decode_params_to_trajectories(best_params)
            for i, traj in enumerate(trajectory_info['trajectories'], 1):
                issues = []
                if traj['t_explode'] >= MISSILE_FLIGHT_TIME:
                    issues.append(f"起爆过晚({traj['t_explode']:.3f}s≥{MISSILE_FLIGHT_TIME:.3f}s)")
                if traj['explode_pos'][2] <= REAL_TARGET_HEIGHT:
                    issues.append(f"高度过低({traj['explode_pos'][2]:.3f}m≤{REAL_TARGET_HEIGHT:.3f}m)")
                
                if issues:
                    print(f"  第{i}枚问题: {', '.join(issues)}")
                else:
                    print(f"  第{i}枚: ✅ 正常")
    else:
        print(f"\n❌ 优化未收敛或未找到有效解")
        print(f"  优化成功标志: {result.success}")
        print(f"  函数值: {result.fun:.8f}")
        print(f"  对应遮蔽时长: {-result.fun:.6f} 秒")
        print(f"  迭代次数: {result.nit}")
        print(f"  函数评估次数: {result.nfev}")
        
        if hasattr(result, 'message'):
            print(f"  优化信息: {result.message}")
        
        # 尝试输出当前最佳解（即使不满足收敛条件）
        if hasattr(result, 'x') and len(result.x) == 8:
            print(f"\n🔍 分析当前最佳候选解:")
            current_best = result.x
            
            # 验证约束
            print(f"约束验证:")
            is_valid = validate_trajectory_constraints(current_best, verbose=True)
            
            if is_valid:
                print(f"✅ 约束满足，但优化未收敛")
            else:
                print(f"❌ 约束不满足")
            
            # 显示参数详情
            trajectory_info = decode_params_to_trajectories(current_best)
            uav_speed, flight_angle, t_drop1, t_delay1, t_drop2, t_delay2, t_drop3, t_delay3 = current_best
            
            print(f"\n参数详情:")
            print(f"  无人机速度: {uav_speed:.4f} m/s")
            print(f"  飞行角度: {flight_angle:.6f} rad ({math.degrees(flight_angle):.2f}°)")
            print(f"  投放序列: 第1枚{t_drop1:.3f}s → 第2枚{t_drop2:.3f}s → 第3枚{t_drop3:.3f}s")
            print(f"  延迟序列: {t_delay1:.3f}s, {t_delay2:.3f}s, {t_delay3:.3f}s")
            
            # 计算起爆时间
            explode_times = trajectory_info['explode_times']
            print(f"  起爆序列: {[f'{t:.3f}s' for t in sorted(explode_times)]}")
            
            # 检查每枚烟幕弹状态
            print(f"\n烟幕弹状态检查:")
            for i, traj in enumerate(trajectory_info['trajectories'], 1):
                status = []
                if traj['t_explode'] >= MISSILE_FLIGHT_TIME:
                    status.append(f"⚠️起爆过晚({traj['t_explode']:.3f}s≥{MISSILE_FLIGHT_TIME:.3f}s)")
                if traj['explode_pos'][2] <= REAL_TARGET_HEIGHT:
                    status.append(f"⚠️高度过低({traj['explode_pos'][2]:.3f}m≤{REAL_TARGET_HEIGHT}m)")
                
                if status:
                    print(f"    第{i}枚: {' '.join(status)}")
                else:
                    print(f"    第{i}枚: ✅ 状态正常")
            
            # 如果有效，计算遮蔽效果
            if is_valid:
                print(f"\n遮蔽效果分析:")
                duration = -three_bombs_objective_function(current_best)
                print(f"  总遮蔽时长: {duration:.6f} 秒")
                
                if duration > 0:
                    debug_time_calculation(current_best, detailed=False)
        else:
            print(f"⚠️  无法获取最佳候选解的详细信息")

"""
check4.py: 问题四验证脚本 (纯Python实现)
输入一组为3架无人机定义的12个参数，利用纯Python计算并输出总遮蔽时长。
"""

import numpy as np
import time
import math

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

# 目标和烟幕参数
REAL_TARGET_CENTER = np.array([0.0, 200.0, 0.0])
REAL_TARGET_RADIUS = 7.0
REAL_TARGET_HEIGHT = 10.0
R_SMOKE = 10.0
SMOKE_DURATION = 20.0
NUM_UAVS = 3
V_SMOKE_SINK_SPEED = 3.0

# 模拟参数
TIME_STEP = 0.01 # 模拟步长

# --- Python 计算函数 ---

def _generate_target_key_points(center, radius, height, num_side=100, num_cap=50):
    """在Python中生成目标圆柱体的关键点"""
    key_points = []
    # 侧面点
    side_heights = np.linspace(0, height, int(np.sqrt(num_side)))
    side_angles = np.linspace(0, 2 * np.pi, int(np.sqrt(num_side)), endpoint=False)
    for h in side_heights:
        for angle in side_angles:
            key_points.append([
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                h
            ])
    # 顶面和底面点
    cap_angles = np.linspace(0, 2 * np.pi, num_cap, endpoint=False)
    for angle in cap_angles:
        # 底面
        key_points.append([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), 0])
        # 顶面
        key_points.append([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), height])
    return np.array(key_points)

TARGET_KEY_POINTS = _generate_target_key_points(REAL_TARGET_CENTER, REAL_TARGET_RADIUS, REAL_TARGET_HEIGHT)

def point_to_line_distance(point, line_start, line_end):
    """计算点到线段的距离"""
    line_vec = line_end - line_start
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq < 1e-12:
        return np.linalg.norm(point - line_start)
    
    point_vec = point - line_start
    projection = np.dot(point_vec, line_vec) / line_len_sq
    
    if projection < 0:
        return np.linalg.norm(point - line_start)
    if projection > 1:
        return np.linalg.norm(point - line_end)
        
    closest_point = line_start + projection * line_vec
    return np.linalg.norm(point - closest_point)

def calculate_total_duration_python(explode_positions, explode_times):
    """
    纯Python实现的总遮蔽时长计算。
    """
    if not explode_positions:
        return 0.0

    num_steps = int(MISSILE_FLIGHT_TIME / TIME_STEP)
    blocked_steps = 0

    for i in range(num_steps):
        t = i * TIME_STEP
        current_missile_pos = P_M1_0 + VEC_V_M1 * t

        # 筛选出当前时刻有效的烟雾云
        active_clouds = []
        for j in range(len(explode_positions)):
            t_explode = explode_times[j]
            if t >= t_explode and t < (t_explode + SMOKE_DURATION):
                time_since_explode = t - t_explode
                cloud_pos = explode_positions[j].copy()
                cloud_pos[2] -= V_SMOKE_SINK_SPEED * time_since_explode
                if cloud_pos[2] > 0: # 烟雾云必须在地面以上
                    active_clouds.append(cloud_pos)
        
        if not active_clouds:
            continue

        # 检查所有关键点是否都被遮蔽
        all_keypoints_blocked = True
        for key_point in TARGET_KEY_POINTS:
            kp_is_blocked = False
            for cloud_pos in active_clouds:
                if point_to_line_distance(cloud_pos, current_missile_pos, key_point) <= R_SMOKE:
                    kp_is_blocked = True
                    break # 这个关键点被挡住了，检查下一个关键点
            
            if not kp_is_blocked:
                all_keypoints_blocked = False
                break # 有一个关键点没被挡住，说明此时未完全遮蔽
        
        if all_keypoints_blocked:
            blocked_steps += 1
            
    return blocked_steps * TIME_STEP

# --- 参数解码函数 ---

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
        uav_speed, flight_angle, t_drop, t_delay = params[i*4 : (i+1)*4]
        uav_initial_pos = UAV_POSITIONS[i]
        uav_direction = calculate_uav_direction_from_angle(flight_angle)
        
        drop_pos = uav_initial_pos + uav_speed * t_drop * uav_direction
        bomb_initial_velocity = uav_speed * uav_direction
        explode_pos = drop_pos + bomb_initial_velocity * t_delay
        explode_pos[2] -= 0.5 * G_ACCEL * t_delay**2
        t_explode = t_drop + t_delay

        trajectories.append({
            'uav_id': i + 1, 'uav_speed': uav_speed, 'flight_angle': flight_angle,
            'explode_pos': explode_pos, 't_explode': t_explode
        })
    return trajectories

if __name__ == "__main__":
    # ==================================================================
    # 在这里输入您要测试的12个参数
    # 格式: [s1, a1, td1, d1, s2, a2, td2, d2, s3, a3, td3, d3]
    # s: 速度, a: 角度(弧度), td: 投放时间, d: 延迟时间
    # ==================================================================
    test_params = [
        95.123426, 0.100862, 0.571370, 0.480708, 137.285929, 5.331952, 7.793024, 4.419031, 134.145844, 4.182773, 7.539341, 6.009883
    ]

    print("🚀 问题四参数验证脚本 (check4.py - 纯Python版)")
    print(f"模拟步长: {TIME_STEP}s, 关键点数量: {len(TARGET_KEY_POINTS)}")
    print("-" * 50)

    # 1. 解码参数
    print("解码输入参数...")
    trajectories = decode_params_to_trajectories(test_params)
    
    explode_positions = []
    explode_times = []
    is_valid = True

    print("\n--- 轨迹详情 ---")
    for traj in trajectories:
        print(f"无人机 FY{traj['uav_id']}:")
        print(f"  飞行参数: 速度 {traj['uav_speed']:.2f} m/s, 角度 {math.degrees(traj['flight_angle']):.2f}°")
        print(f"  起爆时间: {traj['t_explode']:.3f} s")
        print(f"  起爆位置: [{traj['explode_pos'][0]:.1f}, {traj['explode_pos'][1]:.1f}, {traj['explode_pos'][2]:.1f}]")
        
        if traj['t_explode'] >= MISSILE_FLIGHT_TIME or traj['explode_pos'][2] <= REAL_TARGET_HEIGHT:
            is_valid = False
            print("  [警告] 参数违反基本约束 (起爆时间或高度)!")
        
        explode_positions.append(traj['explode_pos'])
        explode_times.append(traj['t_explode'])
    print("-" * 18)

    # 2. 调用Python函数计算
    if not is_valid:
        print("\n⚠️  参数违反基本约束，计算出的遮蔽时长可能为0。")

    print("\n调用Python函数计算总遮蔽时长...")
    start_time = time.time()
    
    total_duration = calculate_total_duration_python(explode_positions, explode_times)
    
    end_time = time.time()
    print(f"计算耗时: {(end_time - start_time) * 1000:.2f} ms")

    # 3. 打印最终结果
    print("\n" + "=" * 50)
    print(f"📊 最终计算结果:")
    print(f"   总遮蔽时长 = {total_duration:.8f} 秒")
    print("=" * 50)
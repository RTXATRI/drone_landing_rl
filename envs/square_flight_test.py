#!/usr/bin/env python3
# 功能：提供独立的 PyBullet 正方形航线飞行演示，用于基础运动学控制验证。
"""
square_flight_test.py
────────────────────────────────────────────────────────────────────────────
无人机基础控制测试：在 3m 高空飞行 2 圈边长为 10m 的正方形。

运行：
    python square_flight_test.py

依赖：
    pip install pybullet numpy
────────────────────────────────────────────────────────────────────────────
"""

import math
import time
import numpy as np
import pybullet as p
import pybullet_data

# ══════════════════════════════════════════════════════════════════════════════
# 参数配置
# ══════════════════════════════════════════════════════════════════════════════
DT              = 0.02          # 控制周期 (s) → 50 Hz
FILTER_ALPHA    = 0.55          # 速度低通滤波系数 (0=无滤波, 1=完全锁定)
MAX_SPEED       = 3.0           # 最大飞行速度 m/s
MAX_YAW_RATE    = 1.5           # 最大偏航角速度 rad/s
ARRIVE_RADIUS   = 0.35          # 到达判定半径 m
ALTITUDE        = 3.0           # 巡航高度 m
SQUARE_SIDE     = 10.0          # 正方形边长 m
LAPS            = 2             # 圈数
TAKEOFF_SPEED   = 1.2           # 起飞速度 m/s
YAW_GAIN        = 2.0           # 偏航跟踪增益

# 正方形四个顶点（逆时针，从原点出发）
WAYPOINTS = np.array([
    [ SQUARE_SIDE / 2,  SQUARE_SIDE / 2, ALTITUDE],
    [-SQUARE_SIDE / 2,  SQUARE_SIDE / 2, ALTITUDE],
    [-SQUARE_SIDE / 2, -SQUARE_SIDE / 2, ALTITUDE],
    [ SQUARE_SIDE / 2, -SQUARE_SIDE / 2, ALTITUDE],
], dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# PyBullet 场景初始化
# ══════════════════════════════════════════════════════════════════════════════

def init_scene():
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.setTimeStep(DT)

    # GUI 外观
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,            0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,        1)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=22,
        cameraYaw=40,
        cameraPitch=-35,
        cameraTargetPosition=[0, 0, 1.5],
    )

    # 地面
    p.loadURDF("plane.urdf")

    # 地面网格辅助线（灰色）
    grid_ext = 8
    for i in range(-grid_ext, grid_ext + 1):
        c = [0.35, 0.35, 0.35]
        p.addUserDebugLine([i, -grid_ext, 0.02], [i,  grid_ext, 0.02], c, 0.5)
        p.addUserDebugLine([-grid_ext, i, 0.02], [ grid_ext, i, 0.02], c, 0.5)

    # 正方形航迹（蓝色虚线，高度 = ALTITUDE）
    sq = np.vstack([WAYPOINTS, WAYPOINTS[0]])   # 首尾相连
    for i in range(len(sq) - 1):
        p.addUserDebugLine(sq[i].tolist(), sq[i+1].tolist(),
                           [0.20, 0.55, 1.0], 2.5)

    # 顶点标记
    for i, wp in enumerate(WAYPOINTS):
        p.addUserDebugText(f"WP{i+1}", wp.tolist(),
                           [1.0, 0.85, 0.0], 1.2)

    # 起飞点标记（原点上方）
    p.addUserDebugLine([0,0,0], [0,0,ALTITUDE], [0.6,0.6,0.6], 1.0,
                       lifeTime=0)

    return client


def build_drone():
    """创建无人机刚体（质量=0，运动学控制）。"""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.22, 0.22, 0.06])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.22, 0.22, 0.06],
                              rgbaColor=[0.15, 0.40, 0.90, 1.0])
    # 四个"旋翼"视觉圆柱
    arm_offsets = [[ 0.22, 0.22], [-0.22, 0.22],
                   [-0.22,-0.22], [ 0.22,-0.22]]
    arm_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.001)  # dummy
    arm_vis = p.createVisualShape(p.GEOM_CYLINDER,
                                  radius=0.12, length=0.02,
                                  rgbaColor=[0.85, 0.85, 0.85, 0.85])
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=[0, 0, 0.1])

    rotors = []
    for ox, oy in arm_offsets:
        r = p.createMultiBody(baseMass=0,
                              baseCollisionShapeIndex=arm_col,
                              baseVisualShapeIndex=arm_vis,
                              basePosition=[ox, oy, 0.1])
        rotors.append(r)

    return body, rotors


def build_landing_pad():
    """原点降落平台（红色）。"""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.04])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.04],
                              rgbaColor=[0.90, 0.25, 0.20, 0.95])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                      baseVisualShapeIndex=vis, basePosition=[0, 0, 0.04])


# ══════════════════════════════════════════════════════════════════════════════
# 运动学控制器
# ══════════════════════════════════════════════════════════════════════════════

class KinematicController:
    def __init__(self):
        self.pos     = np.array([0.0, 0.0, 0.1])
        self.yaw     = 0.0
        self.vel_f   = np.zeros(3)   # filtered velocity
        self.yaw_f   = 0.0

    def compute_cmd(self, target: np.ndarray, speed_limit: float) -> np.ndarray:
        """比例导引：计算指向目标的速度指令。"""
        diff = target - self.pos
        dist = np.linalg.norm(diff)
        if dist < 1e-4:
            return np.zeros(3)
        # 越接近越慢（平滑减速）
        speed = min(speed_limit, dist * 2.5, speed_limit)
        return (diff / dist) * speed

    def compute_yaw_cmd(self, target: np.ndarray) -> float:
        """朝向目标的偏航角速度指令。"""
        dx, dy = target[0] - self.pos[0], target[1] - self.pos[1]
        desired_yaw = math.atan2(dy, dx)
        err = desired_yaw - self.yaw
        # 角度归一化到 (-π, π)
        err = (err + math.pi) % (2 * math.pi) - math.pi
        return float(np.clip(err * YAW_GAIN, -MAX_YAW_RATE, MAX_YAW_RATE))

    def step(self, vel_cmd: np.ndarray, yaw_cmd: float) -> tuple:
        """低通滤波 + 积分。"""
        α = FILTER_ALPHA
        self.vel_f = α * self.vel_f + (1 - α) * vel_cmd
        self.yaw_f = α * self.yaw_f + (1 - α) * yaw_cmd
        self.pos  += self.vel_f * DT
        self.yaw  += self.yaw_f * DT
        self.yaw   = (self.yaw + math.pi) % (2 * math.pi) - math.pi
        return self.pos.copy(), self.yaw


# ══════════════════════════════════════════════════════════════════════════════
# 轨迹追踪线（实时尾迹）
# ══════════════════════════════════════════════════════════════════════════════

class TrailRenderer:
    def __init__(self, color=(1.0, 0.35, 0.10), max_lines=600):
        self.color     = color
        self.max_lines = max_lines
        self._ids      = []
        self._prev     = None

    def update(self, pos: np.ndarray):
        if self._prev is not None:
            lid = p.addUserDebugLine(
                self._prev.tolist(), pos.tolist(),
                self.color, lineWidth=2.0, lifeTime=30.0
            )
            self._ids.append(lid)
            if len(self._ids) > self.max_lines:
                # 移除最旧的线段
                try:
                    p.removeUserDebugItem(self._ids.pop(0))
                except Exception:
                    pass
        self._prev = pos.copy()


# ══════════════════════════════════════════════════════════════════════════════
# HUD：屏幕文字信息
# ══════════════════════════════════════════════════════════════════════════════

class HUD:
    def __init__(self):
        self._ids = {}

    def _put(self, key, text, pos, color=(1,1,1), size=1.1):
        if key in self._ids:
            p.removeUserDebugItem(self._ids[key])
        self._ids[key] = p.addUserDebugText(text, pos, color,
                                             textSize=size, lifeTime=0)

    def update(self, drone_pos, yaw_deg, speed, phase, lap, wp_idx,
               elapsed, dist_to_wp):
        # 状态文字浮在无人机正上方
        above = (drone_pos + np.array([0, 0, 1.2])).tolist()

        self._put("phase",  f"[{phase}]", above, [1.0, 0.92, 0.1], 1.3)

        # 左上角信息面板（固定位置）
        panel_x, panel_y, panel_z = -14, 14, 0.5
        lines = [
            ("pos",   f"Pos  : ({drone_pos[0]:+6.2f}, {drone_pos[1]:+6.2f}, {drone_pos[2]:+6.2f}) m"),
            ("yaw",   f"Yaw  : {yaw_deg:+7.1f} deg"),
            ("speed", f"Speed: {speed:5.2f} m/s"),
            ("lap",   f"Lap  : {lap} / {LAPS}"),
            ("wp",    f"WP   : {wp_idx + 1} / 4"),
            ("dist",  f"Dist : {dist_to_wp:5.2f} m"),
            ("time",  f"Time : {elapsed:6.1f} s"),
        ]
        for i, (key, txt) in enumerate(lines):
            self._put(key, txt,
                      [panel_x, panel_y, panel_z - i * 0.9],
                      [0.9, 0.95, 1.0], 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def sync_bodies(drone_id, rotor_ids, pos, yaw):
    """将 PyBullet 刚体位置/姿态同步到运动学状态。"""
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    p.resetBasePositionAndOrientation(drone_id, pos.tolist(), quat)
    arm_offsets = [[ 0.22, 0.22], [-0.22, 0.22],
                   [-0.22,-0.22], [ 0.22,-0.22]]
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    for rid, (ox, oy) in zip(rotor_ids, arm_offsets):
        rx = pos[0] + ox * cos_y - oy * sin_y
        ry = pos[1] + ox * sin_y + oy * cos_y
        rz = pos[2]
        p.resetBasePositionAndOrientation(rid, [rx, ry, rz], quat)


def main():
    print("=" * 58)
    print("  无人机正方形飞行测试")
    print(f"  边长={SQUARE_SIDE}m  高度={ALTITUDE}m  圈数={LAPS}")
    print(f"  最大速度={MAX_SPEED} m/s  控制频率={int(1/DT)} Hz")
    print("=" * 58)
    print("  关闭 PyBullet 窗口或按 Ctrl+C 退出\n")

    init_scene()
    build_landing_pad()
    drone_id, rotor_ids = build_drone()

    ctrl   = KinematicController()
    trail  = TrailRenderer()
    hud    = HUD()

    # ── 任务状态机 ──────────────────────────────────────────
    phase       = "TAKEOFF"   # TAKEOFF → FLY → LAND → DONE
    wp_idx      = 0
    lap         = 1
    total_wp    = 0            # 已通过的总航点数
    max_wp      = LAPS * len(WAYPOINTS)
    t_start     = time.time()
    step_count  = 0

    takeoff_target = np.array([0.0, 0.0, ALTITUDE])

    while True:
        # ── 窗口关闭检测 ─────────────────────────────────────
        try:
            p.getConnectionInfo()
        except Exception:
            break

        pos = ctrl.pos.copy()
        now = time.time()
        elapsed = now - t_start

        # ── 阶段状态机 ───────────────────────────────────────
        if phase == "TAKEOFF":
            target    = takeoff_target
            vel_cmd   = ctrl.compute_cmd(target, TAKEOFF_SPEED)
            yaw_cmd   = 0.0
            if np.linalg.norm(pos - target) < ARRIVE_RADIUS:
                phase = "FLY"
                print(f"  [t={elapsed:5.1f}s] 起飞完成 → 开始巡航 Lap 1")

        elif phase == "FLY":
            target = WAYPOINTS[wp_idx]
            vel_cmd = ctrl.compute_cmd(target, MAX_SPEED)
            yaw_cmd = ctrl.compute_yaw_cmd(target)

            dist_to_wp = np.linalg.norm(pos - target)
            if dist_to_wp < ARRIVE_RADIUS:
                total_wp += 1
                print(f"  [t={elapsed:5.1f}s] WP{wp_idx+1} 到达  "
                      f"Lap={lap}  总WP={total_wp}/{max_wp}")

                wp_idx = (wp_idx + 1) % len(WAYPOINTS)
                if wp_idx == 0:
                    lap += 1
                    if total_wp >= max_wp:
                        phase = "LAND"
                        print(f"  [t={elapsed:5.1f}s] {LAPS} 圈完成 → 返航降落")
                    else:
                        print(f"  ── Lap {lap} 开始 ──")

        elif phase == "LAND":
            target  = np.array([0.0, 0.0, 0.05])
            vel_cmd = ctrl.compute_cmd(target, 0.8)
            yaw_cmd = ctrl.compute_yaw_cmd(np.array([1.0, 0.0, 0.0]))  # 对准正北
            if pos[2] < 0.15:
                phase = "DONE"
                print(f"  [t={elapsed:5.1f}s] 降落完成！总用时 {elapsed:.1f}s")

        elif phase == "DONE":
            vel_cmd = np.zeros(3)
            yaw_cmd = 0.0
            target  = pos

        # ── 执行控制 ─────────────────────────────────────────
        new_pos, new_yaw = ctrl.step(vel_cmd, yaw_cmd)
        speed = float(np.linalg.norm(ctrl.vel_f))

        sync_bodies(drone_id, rotor_ids, new_pos, new_yaw)
        trail.update(new_pos)

        # ── HUD 更新（每 5 步更新一次，减少 API 调用）───────
        if step_count % 5 == 0:
            dist_to_wp = np.linalg.norm(new_pos - target)
            hud.update(
                drone_pos  = new_pos,
                yaw_deg    = math.degrees(new_yaw),
                speed      = speed,
                phase      = phase,
                lap        = lap,
                wp_idx     = wp_idx,
                elapsed    = elapsed,
                dist_to_wp = dist_to_wp,
            )

        p.stepSimulation()
        step_count += 1

        # ── 实时速率控制（贴近真实 DT）───────────────────────
        time.sleep(max(0, DT - (time.time() - now)))

        if phase == "DONE" and step_count > 150:
            print("  仿真结束，窗口将在 3 秒后关闭…")
            time.sleep(3)
            break

    p.disconnect()
    print("  已退出。")


if __name__ == "__main__":
    main()

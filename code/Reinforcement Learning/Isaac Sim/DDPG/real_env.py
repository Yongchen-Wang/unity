import gym
from gym import spaces
import numpy as np
from sensapex import UMP
import cv2

class RealWorldEnv(gym.Env):
    # metadata = {"render.modes": ["human"]}

    def __init__(self, resolution=256):
        super().__init__()
        self.resolution = resolution
        self.step_num = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=0.5, shape=(4,), dtype=np.float32)  # 两个物体的二维坐标

        # 初始化真实环境的硬件或传感器
        self.initialize_real_world()

    def initialize_real_world(self):
        """
        初始化真实环境中的硬件和传感器。
        """
        # 获取UMP实例和特定的操纵杆设备
        self.ump = UMP.get_ump()
        self.manipulator = self.ump.get_device(1)

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)

    def find_ball_position(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([19, 12, 47])
        upper_color = np.array([55, 78, 87])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ball_contour = None
        max_circularity = 0

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if area < 10 or perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if circularity > 0.7 and circularity > max_circularity:
                ball_contour = contour
                max_circularity = circularity

        if ball_contour is not None:
            (x, y), radius = cv2.minEnclosingCircle(ball_contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            x1 = int(x - radius)
            y1 = int(y - radius)
            x2 = int(x + radius)
            y2 = int(y + radius)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return frame, center
        return frame, None

    def pixel_to_space(self, x_pixel, y_pixel, x1, y1, x2, y2, space1, space2):
        x_space1, y_space1 = space1
        x_space2, y_space2 = space2
        x_space = x_space1 + (x_pixel - x1) * (x_space2 - x_space1) / (x2 - x1)
        y_space = y_space1 + (y_pixel - y1) * (y_space2 - y_space1) / (y2 - y1)
        return y_space, x_space

    def get_real_world_observation(self):
        """
        从真实环境中获取观测。
        """
        # 获取agent（机器人末端）的位置
        agent_position = np.array(self.manipulator.get_pos()) / 10000.0  # 将单位转换成适合的比例

        # 获取target（通过相机获取）的位置信息
        ret, frame = self.cap.read()  # 从摄像头读取图像
        if not ret:
            return agent_position[:2], np.array([0.25, 0.25])  # 如果没有获取到图像，返回默认值

        frame_with_ball, position = self.find_ball_position(frame)
        cv2.imshow('Frame with Ball', frame_with_ball)

        if position is not None:
            x1, y1 = 115, 25
            x2, y2 = 554, 455
            space1 = (0, 0)
            space2 = (0.36, 0.36)
            space_position = self.pixel_to_space(position[0], position[1], x1, y1, x2, y2, space1, space2)
            return agent_position[:2], np.array(space_position)
        return agent_position[:2], np.array([0.25, 0.25])  # 如果没有找到目标，返回默认值


    def send_action_to_real_world(self, action):
        """
        将动作发送到真实环境中执行。
        """
        current_pos = np.array(self.manipulator.get_pos())

        deltax, deltay = action
        deltax *= 4000
        deltay *= 6000

        new_x = np.clip(current_pos[0] + deltax, self.min_limit, self.max_limit)
        new_y = np.clip(current_pos[1] + deltay, self.min_limit, self.max_limit)
        new_z = current_pos[2]  # 保持z轴不变

        self.manipulator.goto_pos((new_x, new_y, new_z, 10000), 5000)
        print(f"Moved to new position: {(new_x, new_y, new_z)}")

    def calculate_reward(self, current_target_position, current_agent_position, action_success):
        done = False
        info = True
        reward = -1  # 每一步都要扣一分
        des = np.array([0.34, 0.05, 0.035])
        norm_endpoint = np.linalg.norm(current_target_position - des)

        if not action_success:
            reward -= 10

        if norm_endpoint != 0:
            reward += (1 - 20 * norm_endpoint)

        if 0.3 <= current_target_position[0] <= 0.34 and current_target_position[1] <= 0.07:
            done = True
            reward += 1000
            print("Done")

        return done, reward, info

    def step(self, action):
        self.step_num += 1

        # 将动作发送到真实环境中
        self.send_action_to_real_world(action)

        # 获取新的观测
        current_target_position, current_agent_position = self.get_real_world_observation()

        # 计算奖励
        done, reward, info = self.calculate_reward(current_target_position, current_agent_position, True)  # 假设action_success为True
        state = np.concatenate([np.round(current_agent_position[:2], 3), np.round(current_target_position[:2], 3)])  # 两个二维坐标拼接成四维状态

        return state, reward, done, info

    # def render(self, mode="human"):
    #     # 渲染真实环境（如果需要）
    #     pass

    def close(self):
        # 关闭真实环境的连接
        self.cap.release()
        cv2.destroyAllWindows()

    def seed(self, seed=None):
        pass

if __name__ == "__main__":
    env = RealWorldEnv()
    model = load_your_trained_model()  # 加载训练好的模型

    max_steps = 512
    agent_position, target_position = env.get_real_world_observation()
    state = np.concatenate([np.round(agent_position[:2], 3), np.round(target_position[:2], 3)])  # 初始状态
    
    for i in range(max_steps):
        action = model.predict(state)  # 使用模型预测动作
        state, reward, done, info = env.step(action)
        print(f"Step {i}, State: {state}, Reward: {reward}, Done: {done}")
        if done:
            break
    env.close()

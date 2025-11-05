import random
from collections import deque

from utils import *
from parameters import *
# -----------------------------
# 通信系统
# -----------------------------

class Communication:
    """简单的点对点基于距离的通信系统，带可配置丢包与延迟（延迟目前同步模拟）"""
    def __init__(self, packet_loss=COMM_PACKET_LOSS, delay=COMM_DELAY):
        self.packet_loss = packet_loss
        self.delay = delay
        self.queue = deque()  # (deliver_time, sender_id, receiver_id, message)

    def send(self, sender, receiver, message, now_time):
        """发送消息到单个接收者；不保证送达（丢包）"""
        if random.random() < self.packet_loss:
            return False
        deliver_time = now_time + self.delay
        self.queue.append((deliver_time, sender, receiver, message))
        return True

    def broadcast(self, sender, receivers,\
        message, now_time, \
        range_limit=None):
        sent = 0
        for r in receivers:
            if range_limit is None or distance(sender.pos, r.pos) <= range_limit:
                if self.send(sender, r, message, now_time):
                    sent += 1
        return sent

    def deliver(self, now_time):
        """将已到达时间的消息投递给目标（外部负责调用并处理）"""
        delivered = []
        while self.queue and self.queue[0][0] <= now_time:
            _, sender, receiver, message = self.queue.popleft()
            delivered.append((sender, receiver, message))
        return delivered
    


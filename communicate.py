# import random
# from collections import deque

# from utils import *
# from parameters import *
# # -----------------------------
# # 通信系统
# # -----------------------------

# class Communication:
#     """简单的点对点基于距离的通信系统，带可配置丢包与延迟（延迟目前同步模拟）"""
#     def __init__(self, packet_loss=COMM_PACKET_LOSS, delay=COMM_DELAY):
#         self.packet_loss = packet_loss
#         self.delay = delay
#         self.queue = deque()  # (deliver_time, sender_id, receiver_id, message)

#     def send(self, sender, receiver, message, now_time):
#         """发送消息到单个接收者；不保证送达（丢包）"""
#         if random.random() < self.packet_loss:
#             return False
#         deliver_time = now_time + self.delay
#         self.queue.append((deliver_time, sender, receiver, message))
#         return True

#     def broadcast(self, sender, receivers,\
#         message, now_time, \
#         range_limit=None):
#         sent = 0
#         for r in receivers:
#             if range_limit is None or distance(sender.pos, r.pos) <= range_limit:
#                 if self.send(sender, r, message, now_time):
#                     sent += 1
#         return sent

#     def deliver(self, now_time):
#         """将已到达时间的消息投递给目标（外部负责调用并处理）"""
#         delivered = []
#         while self.queue and self.queue[0][0] <= now_time:
#             _, sender, receiver, message = self.queue.popleft()
#             delivered.append((sender, receiver, message))
#         return delivered
    

import random
from collections import deque

from utils import *
from parameters import *


class Communication:
    """
    Hierarchical multi-robot communication system with:
    - structured messaging
    - delay & packet loss
    - enforced parent-child routing rules (L ↔ M ↔ S)
    - distance-based reachability (optional)
    """

    def __init__(self,
                 packet_loss=COMM_PACKET_LOSS,
                 delay=COMM_DELAY,
                 distance_limit=AGENT_COMM_RANGE):
        self.packet_loss = packet_loss
        self.delay = delay
        self.distance_limit = distance_limit
        self.queue = deque()   # (deliver_time, sender, receiver, message)

    # ================================================================
    #  Communication rules (core hierarchical constraints)
    # ================================================================
    def can_communicate(self, sender, receiver):
        """Enforce the hierarchical rule set:
            S → parent M
            M → parent L
            L → all M
            M → own S children
            S/M → ANY L only for VICTIM_ALERT or DEATH_ALERT
        """

        # 1. Same robot? ignore
        if sender.id == receiver.id:
            return False

        # 2. Category detection
        SENDER_S = not sender.is_large
        SENDER_M = sender.is_large and (not getattr(sender, "is_brain", False))
        SENDER_L = getattr(sender, "is_brain", False)

        RECEIVER_S = not receiver.is_large
        RECEIVER_M = receiver.is_large and (not getattr(receiver, "is_brain", False))
        RECEIVER_L = getattr(sender, "is_brain", False)

        # ---------------------------
        # Small → Middle (only parent)
        # ---------------------------
        if SENDER_S and RECEIVER_M:
            return (sender.father_id == receiver.id)

        # ---------------------------
        # Middle → Small (children only)
        # ---------------------------
        if SENDER_M and RECEIVER_S:
            return (receiver.father_id == sender.id)

        # ---------------------------
        # Middle → Large (parent only)
        # ---------------------------
        if SENDER_M and RECEIVER_L:
            return (sender.father_id == receiver.id)

        # ---------------------------
        # Large → Middle (all large → all middle allowed)
        # ---------------------------
        if SENDER_L and RECEIVER_M:
            return True

        # ---------------------------
        # Small/Middle → ANY Large only in emergency (DEATH or VICTIM)
        # ---------------------------
        if RECEIVER_L and (SENDER_S or SENDER_M):
            # Only emergency messages allowed directly
            # If trying to send non-emergency → block
            return False

        return False

    # ================================================================
    #  Low-level message sending
    # ================================================================
    def send(self, sender, receiver, message, now_time):
        """Send message from sender to receiver using hierarchy & distance constraints."""

        # 1. Packet loss
        if random.random() < self.packet_loss:
            return False

        # 2. Range limit
        if self.distance_limit is not None:
            if distance(sender.pos, receiver.pos) > self.distance_limit:
                return False

        # 3. Hierarchical communication rules
        if not self.can_communicate(sender, receiver):
            return False

        # 4. Deliver with delay
        deliver_time = now_time + self.delay
        self.queue.append((deliver_time, sender, receiver, message))
        return True

    # ================================================================
    #  Broadcast utility
    # ================================================================
    def broadcast(self, sender, receivers, message, now_time):
        """Broadcast a message to a list of receivers."""
        count = 0
        for r in receivers:
            if self.send(sender, r, message, now_time):
                count += 1
        return count

    # ================================================================
    #  Deliver messages
    # ================================================================
    def deliver(self, now_time):
        """Return all messages whose delay time has reached."""
        delivered = []
        while self.queue and self.queue[0][0] <= now_time:
            _, sender, receiver, message = self.queue.popleft()
            delivered.append((sender, receiver, message))
        return delivered


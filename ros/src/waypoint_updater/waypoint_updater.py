#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from tf.transformations import euler_from_quaternion

import math
import time

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        rospy.Subscriber('/traffic_waypoint',Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity',TwistStamped, self.velocity_cb)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        self.rate = rospy.Rate(50)

        self.ego_pose = None

        self.waypoints = None
        self.next_waypoint = None

        self.tl_waypoint = 0
        self.ego_vel = None

        self.stopping = False

        ref_vel = 11.11111

        while (self.waypoints is None) or (self.ego_pose is None):
            time.sleep(0.05)

        # Publish the final waypoints.
        while not rospy.is_shutdown():
            self.next_waypoint = self.get_next_waypoint()
            vel = self.ego_vel #self.get_waypoint_velocity(self.waypoints.waypoints[self.next_waypoint])
            if self.tl_waypoint > 0:
                dist1 = self.get_distance(self.next_waypoint, self.tl_waypoint)
                if self.stopping == False and dist1 < 100:
                    self.stopping = True
                    vel = self.ego_vel #self.get_waypoint_velocity(self.waypoints.waypoints[self.next_waypoint])
                    nr_wps = self.tl_waypoint - self.next_waypoint
                    if nr_wps > 0: #to avoid division by zero
                        slowdown = vel / float(nr_wps)
                    else:
                        slowdown = 0
                    for wp in self.waypoints.waypoints[self.next_waypoint:self.tl_waypoint]:
                        vel -= slowdown
                        #vel = -10
                        if vel < .5:
                            vel = 0
                        wp.twist.twist.linear.x = vel
                        #rospy.loginfo(vel)
                if self.stopping == True and self.ego_vel < 2.0: #Added creep feat
                    if dist1 > 5:
                        vel = 0.9
                        for wp in self.waypoints.waypoints[self.next_waypoint:self.tl_waypoint]:
                            wp.twist.twist.linear.x = vel
                    else:
                        vel = 0.
                        for wp in self.waypoints.waypoints[self.next_waypoint:self.tl_waypoint]:
                            wp.twist.twist.linear.x = vel
                    #rospy.loginfo(dist1)

            if self.tl_waypoint < 0 and self.stopping == True:
                self.stopping = False
                for wp in range(len(self.waypoints.waypoints)):
                    self.set_waypoint_velocity(self.waypoints.waypoints,wp,ref_vel)
                #rospy.loginfo(ref_vel)

            final_waypoints = Lane()
            final_waypoints.waypoints = self.waypoints.waypoints[self.next_waypoint:self.next_waypoint+LOOKAHEAD_WPS]
            self.final_waypoints_pub.publish(final_waypoints)
            self.rate.sleep()

    def velocity_cb(self, msg):
        self.ego_vel = msg.twist.linear.x

    def pose_cb(self, msg):
        self.ego_pose = msg.pose

    def waypoints_cb(self, waypoints):
        rospy.loginfo("waypoints dtype: %s", type(waypoints))
        rospy.loginfo("waypoints.waypoints[0] dtype: %s", type(waypoints.waypoints[0]))
        self.waypoints = waypoints
        self.sub2.unregister()

    def traffic_cb(self, msg):
        self.tl_waypoint = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def get_distance(self, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(self.waypoints.waypoints[wp1].pose.pose.position, self.waypoints.waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def get_l2_distance(self, wp):
        '''
        Computes the distance between the ego car and the given waypoint.
        '''
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        return dl(self.waypoints.waypoints[wp].pose.pose.position, self.ego_pose.position)

    def get_closest_waypoint(self, begin=0, end=None):
        '''
        Returns the waypoint that is closest to the ego car.
        '''
        closest_waypoint = None
        closest_waypoint_dist = 1000000
        if end is None:
            end = len(self.waypoints.waypoints)

        if end < begin: # Wrap around after the last waypoint.
            for i in range(begin, len(self.waypoints.waypoints)):
                dist = self.get_l2_distance(i)
                if dist < closest_waypoint_dist:
                    closest_waypoint = i
                    closest_waypoint_dist = dist
            for i in range(0, end):
                dist = self.get_l2_distance(i)
                if dist < closest_waypoint_dist:
                    closest_waypoint = i
                    closest_waypoint_dist = dist
        else:
            for i in range(begin, end):
                dist = self.get_l2_distance(i)
                if dist < closest_waypoint_dist:
                    closest_waypoint = i
                    closest_waypoint_dist = dist
        return closest_waypoint

    def get_next_waypoint(self):
        if self.next_waypoint is None: # If this is the first time we're computing the next waypoint, we have to iterate over all waypoints.
            closest_waypoint = self.get_closest_waypoint()
        else:
            closest_waypoint = self.get_closest_waypoint(begin=self.next_waypoint, end=((self.next_waypoint + 100) % len(self.waypoints.waypoints)))
        # Check whether the closest waypoint is ahead of the ego car or behind it.
        if self.is_ahead(closest_waypoint):
            # If it is ahead, that's our guy.
            return closest_waypoint
        else:
            # If not, then the next waypoint after it must be our guy.
            if (closest_waypoint + 1) < len(self.waypoints.waypoints):
                return closest_waypoint + 1
            else: # Wrap around after the last waypoint.
                return 0

    def is_ahead(self, index):
        '''
        Returns `True` if the waypoint that `index` references lies ahead of the car and `False` otherwise.
        '''
        # Get the ego car's orientation quaternion...
        quaternion = (self.ego_pose.orientation.x,
                      self.ego_pose.orientation.y,
                      self.ego_pose.orientation.z,
                      self.ego_pose.orientation.w)
        # ...and compute the yaw from the quaternion.
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        # Compute the angle of the waypoint relative to the ego car's heading.
        dx = self.waypoints.waypoints[index].pose.pose.position.x - self.ego_pose.position.x
        dy = self.waypoints.waypoints[index].pose.pose.position.y - self.ego_pose.position.y

        waypoint_angle = math.atan2(dy, dx)

        diff_angle = abs(yaw - waypoint_angle)
        if diff_angle > math.pi / 4:
            return False
        else:
            return True

if __name__ == '__main__':
    try:
        waypoint_updater = WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

What is the difference between a computer board vs flight controller?

Compute board: used for computing stuff and commanding the drone to do things (state machine -> certain result, a matching response to the result); voluntary actions

Flight controller: involuntary stuff that the drone doesn’t need to control; responsible for steady flight and uses the drone’s sensors

What operating systems, software, firmware, and middleware do each run?
OS: Linux on board, NuttX on controller
Software: Mavlink (both)
Firmware: PX4
Middleware: ROS2

How do the two communicate with each other?

Through MAVROS2

Which sensors are processed by which processor? 

IMU, GPS, Sensors are controlled by MAVLINK

Cameras and Perception Node sensors like Lidar are controlled by ROS2.

What are NuttX, uORB and Mavlink?

NuttX is a real-time operating system. uORB (Micro Object Request Broker) is a messaging system used within the PX4 flight stack. It plays a crucial role in the communication infrastructure of the PX4 firmware, facilitating the exchange of data between different modules and components within the flight controller. Mavlink is a messaging protocol for communicating with drones.
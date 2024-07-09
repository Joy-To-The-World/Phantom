What are some of the difference between the a research and development drone vs other "commercial" or "toy" drones?

The flight computer in an R&D drone is more powerful, since it needs to run more complex code.  It is also more specialized to the specific purpose of the R&D, rather than commercial drones which are typically more general.

What are some current applications of autonomous drones? Can you think of any future applications as technology improves (e.g. faster, smaller, more efficient computers)?

Current applications include package delivery, search and rescue, agriculture (spreading pesticides on a crop field), and photography. In the future, drones can be used for saving people (people who suffered a stroke -> drones can be used to do CPR), clean windows of very tall buildings, building skyscrapers, and autonomous airplanes (without human pilots; self-driving airplanes).

Describe the difference between the Compute Board and the Flight Controller. What purposes do each serve? What operating systems do each run?

The compute board is used to command the drone to do certain actions (it is voluntary and uses the state machine to generate proper responses to certain stimuli). Meanwhile, the flight controller is used to ensure the drone is set up to use the autopilot or the compute board (“involuntary actions” including making sure the drone is steady when in flight and controls the drone’s sensors).

Which communication architecture are we using to connect are computers to the drone: Peer2Peer or centralized? What about the remote control - drone communication?

The connection between the computers and drone is centralized and the remote control-drone connection is Peer2Peer.

True or False: For manual flight control, the remote control communicates with the drone over wifi.

False, the remote control and drone communicate using other means.

In order to know where the drone is in the world, it needs some form of positioning sensor/algorithms, otherwise it would be flying blind. What are the different types of positioning sensors available? Which do you think we are going to use during the class?

Positioning sensors include the IMU, GPS, and LiDAR.  I think we will be using the IMU or GPS.

True or False: during our indoor flights, we can use the GPS sensor to estimate the drone's position in the world.

False: You would need to know the coordinates of the corners of the room, but if you did, then this is true.

Are optical flow algorithms responsible for mapping the environment? If not, can you describe conceptually what optical flow does?

Optical flow is responsible for mapping the movement of objects in the camera’s range.  So this is partially true.

Which potential sensors on a Drone enables 3D visual mapping?

The 3D camera, and potentially the 3D LiDAR let the drone see its surrounding environment.

How does the Compute Board communicate with the Flight Controller?

The Compute Board uses MAVLink to give the Flight Controller basic instructions based on the code it is running.

What is PX4 and where is it running on the drone?

PX4 is an autopilot software in which you don’t need to share your source code and it runs on the flight controller.

Which of these best describes MAVLink: 1. an operating system used on the drone, 2. a sensor on the drone, 3. a communication protocol on the drone, 4. a programming language

3. Mavlink is a communication protocol on the drone.

If I want to write a new, complex computer vision algorithm for the drone, should I add it to the Flight Controller firmware? if not, where should I add it and why?

No, you would add it to the computer board, since it has more computing power and would thus be capable of running the computer vision algorithm without slowdowns or crashes.
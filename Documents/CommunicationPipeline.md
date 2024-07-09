Set 1:

List all rosnodes that exist after running the publisher
/talker and /listener.

List all rostopics that are being present after running the publisher
/chatter.

What command do I run to see what is being published to a topic?
rostopic echo /topic_name

What would I change in the publisher to make it publish messages more frequently?
Add spin to the code inside the publisher, specifically rclpy.spin(node).

In the terminal I run the python script for the subscriber and see information being printed. Does this mean it is publishing to a topic?
No, because publishing a topic also needs communication with the subscriber node, to check if the topic is published, we need to use rostopic echo to see if its published.

What is the main benefit of a composed node? How might this help in drone autonomy applications?
The main benefit of a composed node is that it can manage multiple nodes and through using them to communicate can coordinate all the different functions/actions of a drone, like feedback control, flight, recording, identifying obstacles etc.. This can greatly help the drone in organizing its actions and coordinating itself.


Set 2:


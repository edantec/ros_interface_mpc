# Start time of the trajectory
builtin_interfaces/Time stamp

std_msgs/Float64MultiArray xs
std_msgs/Float64MultiArray us
std_msgs/Float64MultiArray k0

# For kinodynamics
std_msgs/Int8MultiArray contact_states
std_msgs/Float64MultiArray forces
std_msgs/Float64MultiArray ddqs

# For debugging/logging purpose
float64 process_duration
# Start time of the trajectory
builtin_interfaces/Time stamp

std_msgs/Float64MultiArray xs
std_msgs/Float64MultiArray us
std_msgs/Float64MultiArray k0

# For kinodynamics
bool[] contact_states
float64[] forces
float64[] a0

# For debugging/logging purpose
float64 process_duration
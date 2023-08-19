Data (csv format) is organized in folders first by task then by trial. Three trial folders are included for each of the following tasks:
- raster: wiping the surface of a bamboo cutting board side to side as the hand  travels from the bottom ot the top or top to bottom
- circle: wiping the surface of a bamboo cutting board in small counterclockwise circles as the hand covers travels around different parts in a general counterclockwise trajectory.
- spiral: wiping the surface of a bamboo cutting board starting in its center and spiraling ountward, then spiraling inwards once coming closer to the edge (without reaching it)
- edgeC: following the edge of a round plastic bowl, with half the sensor making contact
- edgeR: following the edge of a rectangular bamboo cutting board, with half the sensor making contact
Each trial is a recording of approximately 55 continuous seconds.

 Each trial folder contains data files corresponding to the following tasks:
- wipe_{task}_mocap_hand: motion capture (mocap) data for the hand that’s holding the sensor (3D coordinates of 8 markers = 16 columns). Columns are grouped by threes (i.e. x-y-z, x-y-z,...)
- wipe_{task}_mocap_object: mocap data for the object (16 columns as above)
- wipe_{task}_sensor: comprehensive tactile sensor data (4 deflection directions of 9 taxels = 36 columns)
- wipe_{task}_sensor_comb: tactile sensor data grouped along different directions  (4 cardinal directions, 2 rotational (clockwise and counterclockwise), and overall pressure = 7 columns)
- wipe_{task}_sensor_partial_comb: portions of the taxels are combined, to differentiate between differentiate partial contacts/interactions

Each data file includes column labels (first row) and data (rest of the rows). Each row is a frame (sample) of data, sampled at 60Hz.

Columns headings for the sensor data are defined as follows:
- wipe_{task}_sensor: taxel_#_left,taxel_#_up,taxel_#_right,taxel_#_down, where # ranges from 1 to 9. Taxels are numbered with 1 being the bottom left, 2 is bottom center, 3 is bottom right, moving up to 9 being top right.
- wipe_{task}_sensor_comb: [taxels_up,taxels_down,taxels_right,taxels_left,taxels_clockwise,taxels_counterclockwise,taxels_pressure]. See Fig. 5 in https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8990001 for a visual explanation of how the taxels are summed for the linear and rotational columns.
- wipe_{task}_sensor_partial_comb: [taxel_left_up] corresponds to measuring the upward deflection of the left three taxels. Another example, [taxel_down_right] corresponds to measuring rightward deflection of the bottom three taxels. [taxels_{side}_pressure] corresponds to the overall pressure (sum of all deflection directions) of the corresponding sensor portion (defined by side: up/down/left/right).

Columni headings for the mocap data is in the following order:
- [marker_#_x,marker_#_y,marker_#_z,marker_#_x,...] where # ranges from 1 to 8 (markers).
import numpy as np
import rosbag
import sys
import rospy
import tf

def read_frames(bag, start_time=400, end_time=410):


	ts = []
	frames = ['vicon', 'vicon_wrist1','vicon_chest1','vicon_wrist2','vicon_chest2'] + ['%d' % i for i in range(24)]
	# q = {s: [] for s in frames}
	# x = {s: [] for s in frames}
	time_buffer = {s: [] for s in frames}

	demo_buffer = {s:{'q':[], 'x':[], 't':[]} for s in frames}
	demo_buffer['IMU'] = {'data':[], 't':[]}
	demo_buffer['force'] = {'f':[],'tau':[], 't':[]}

	t0 = None

	frames_det = []

	# find t0
	for topic, m, t in bag:
		t0 = t.to_sec()
		break


	for topic, m, t in bag.read_messages(topics=['/tf']):
		transf = m.transforms[0]
		if transf.child_frame_id == 'vicon':
			q_ = transf.transform.rotation
			x_ = transf.transform.translation

			trans_mat = tf.transformations.translation_matrix([x_.x, x_.y, x_.z])
			rot_mat = tf.transformations.quaternion_matrix([q_.x, q_.y, q_.z, q_.w])

			vicon_mat = np.dot(trans_mat, rot_mat)
			vicon_inv_mat = np.linalg.pinv(vicon_mat)

			break

	for topic, m, t in bag.read_messages(topics=['/tf','/IMUdata', '/iitft_driver_node/ft'],
										 start_time=rospy.Time(start_time+t0), end_time=rospy.Time(end_time+t0)):

		if topic == '/tf':
			transf = m.transforms[0]
			frame_id = transf.child_frame_id

			# if frame_id == '22':

			if frame_id not in frames_det:
				frames_det += [frame_id]

			if frame_id in frames:
				q_ = transf.transform.rotation
				x_ = transf.transform.translation

				if transf.header.frame_id != 'world':
					t_mat_ = tf.transformations.translation_matrix([x_.x, x_.y, x_.z])
					r_mat_ = tf.transformations.quaternion_matrix([q_.x, q_.y, q_.z, q_.w])

					m_mat_ = np.dot(t_mat_, r_mat_)
					m_t_mat = np.dot(vicon_mat,m_mat_)
					# m_t_mat = np.dot(vicon_inv_mat, m_mat_)

					x_.x, x_.y, x_.z = tf.transformations.translation_from_matrix(m_t_mat)
					q_.x, q_.y, q_.z, q_.w = tf.transformations.quaternion_from_matrix(m_t_mat)

				# sys.stdout.write(".")
				# import pdb; pdb.set_trace()
				# q[frame_id] += [[q_.x, q_.y, q_.z, q_.w]]
				# x[frame_id] += [[x_.x, x_.y, x_.z]]

				demo_buffer[frame_id]['q'] += [[q_.x, q_.y, q_.z, q_.w]]
				demo_buffer[frame_id]['x'] += [[x_.x, x_.y, x_.z]]
				demo_buffer[frame_id]['t'] += [t.to_sec()-t0]

				time_buffer[frame_id] += [t.to_sec()]

		elif topic == '/IMUdata':
			demo_buffer['IMU']['t'] += [t.to_sec()-t0]
			demo_buffer['IMU']['data'] += [m.data]

		elif topic == '/iitft_driver_node/ft':
			demo_buffer['force']['t'] += [t.to_sec()-t0]

			f_ = m.wrench.force
			tau_ = m.wrench.torque

			demo_buffer['force']['tau'] += [[tau_.x, tau_.y, tau_.z]]
			demo_buffer['force']['f'] += [[f_.x, f_.y, f_.z]]

	for topic in demo_buffer.keys():
		for feature in demo_buffer[topic].keys():
			demo_buffer[topic][feature] = np.asarray(demo_buffer[topic][feature])

	# print frames_det
	return demo_buffer
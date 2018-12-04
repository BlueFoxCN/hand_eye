%ISHOMOG	test if argument is a homogeneous transformation (4x4)


function h = is_homog(tr)
	h = all(size(tr) == [4 4]);
"""
This is to sanity check the DCM and QTR functions in ps7.py, but used
elsewhere in the code.  This is not a complete test of the functions.
"""

import numpy as np

import source.attitudes as att
import source.rotation as rot


num_tests = 10

for i in range(num_tests):
    np.random.seed(i)
    print("--- Test ", i)

    # we need 6 random numbers between -pi and pi
    rng_ang = np.random.uniform(-np.pi, np.pi, 6)

    dcmBN = rot.dcmX(rng_ang[0]) @ rot.dcmZ(rng_ang[1]) @ rot.dcmY(rng_ang[2])
    dcmRN = rot.dcmX(rng_ang[3]) @ rot.dcmZ(rng_ang[4]) @ rot.dcmY(rng_ang[5])
    dcmBR = dcmRN.T @ dcmBN

    qtrBR = att.QTR( dcm = dcmBR )
    qtrBN = att.QTR( dcm = dcmBN )
    qtrRN = att.QTR( dcm = dcmRN )

    mrpBR = att.MRP( dcm = dcmBR )
    mrpBN = att.MRP( dcm = dcmBN )
    mrpRN = att.MRP( dcm = dcmRN )

    qtrBNoverRN = qtrBN / qtrRN
    # Convert to numpy to take the difference
    qtrBNoverRN_np = np.array(qtrBNoverRN.qtr)
    qtrBR_np = np.array(qtrBR.qtr)
    diff = qtrBNoverRN_np - qtrBR_np
    diff_mag = np.linalg.norm(diff)

    if diff_mag > 1e-6:
        print("Too large of a difference! ")
        print("qtrBNoverRN = ", qtrBNoverRN)
        print("qtrBR       = ", qtrBR)
        print("diff        = ", diff)
        print("diff_mag    = ", diff_mag)
    else:
        print("Quaternion test passed!")

    mrpBNoverRN = mrpBN / mrpRN
    # Convert to numpy to take the difference
    mrpBNoverRN_np = np.array(mrpBNoverRN.mrp)
    mrpBR_np = np.array(mrpBR.mrp)
    diff = mrpBNoverRN_np - mrpBR_np
    diff_mag = np.linalg.norm(diff)

    if diff_mag > 1e-6:
        print("Too large of a difference! ")
        print("mrpBNoverRN = ", mrpBNoverRN)
        print("mrpBR       = ", mrpBR)
        print("diff        = ", diff)
        print("diff_mag    = ", diff_mag)
    else:
        print("MRP test passed!")
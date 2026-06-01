from utils import soa as SOA
from utils import objects as ob
import numpy as np

joint = ob.SphericalJoint()
link = ob.Link(mass=100.0, l_hinge=np.array([0, 0, 1]), joint=joint)


V_test = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0])  # ω and v both nonzero
result = V_test @ SOA.spatialskewbar(V_test) @ link.M @ V_test
print(result)  # Must be 0.0
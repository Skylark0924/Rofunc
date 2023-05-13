## URDF Models

The human body is modeled as a rigid multi-body system with a certain
 number of links connected by joints.  The model consists of simple geometric
 shapes (parallelepiped, cylinder, sphere) whose dimensions are dependent 
from the Xsens motion acquisition by making the model dimensions 'scalable'
 for different subjects.  Due to the strong dependency from the Xsens model
 (23-links biomechanical model), we reconstruct our model in a very similar
 way for a better matching with the data coming from the Xsens motion tracking. 

### Model without articulated hands

- `XSensModelStyle_48URDFtemplate.urdf`
  -   48 revolute 1-DoF joints;
- `XSensModelStyle_66URDFtemplate.urdf`
  -   66 revolute 1-DoF joints;

 <img src="https://user-images.githubusercontent.com/10923418/70060897-a0566f00-15e3-11ea-8abb-6d0d0dd38553.png" width="650" height="700">

### Model with aurticulated hands

- `XSensModelStyle_48URDFhands_template.urdf`
  -   48 revolute 1-DoF joints;
  -   21 revolute 1-DoF joints for the right hand;
  -   21 revolute 1-DoF joints for the left hand;

 <img src="https://user-images.githubusercontent.com/10923418/70060904-a2203280-15e3-11ea-92a8-f56b02ccdbc4.png" width="650" height="700">

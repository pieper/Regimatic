from __main__ import vtk, qt, ctk, slicer
import numpy

#
# Regimatic
#

class Regimatic:
  def __init__(self, parent):
    parent.title = "Regimatic"
    parent.categories = ["Registration"]
    parent.dependencies = []
    parent.contributors = ["Steve Pieper (Isomics)"] # replace with "Firstname Lastname (Org)"
    parent.helpText = """
    Steerable registration example as a scripted loadable extension.
    """
    parent.acknowledgementText = """
    This file was originally developed by Steve Pieper and was partially funded by NIH grant 3P41RR013218.
""" # replace with organization, grant and thanks.
    self.parent = parent

#
# qRegimaticWidget
#

class RegimaticWidget:
  def __init__(self, parent = None):
    if not parent:
      self.parent = slicer.qMRMLWidget()
      self.parent.setLayout(qt.QVBoxLayout())
      self.parent.setMRMLScene(slicer.mrmlScene)
    else:
      self.parent = parent
    self.layout = self.parent.layout()
    if not parent:
      self.setup()
      self.parent.show()

    self.logic = RegimaticLogic()

  def setup(self):
    # Instantiate and connect widgets ...

    # reload button
    self.reloadButton = qt.QPushButton("Reload")
    self.reloadButton.toolTip = "Reload this module."
    self.reloadButton.name = "Regimatic Reload"
    self.layout.addWidget(self.reloadButton)
    self.reloadButton.connect('clicked()', self.onReload)

    #
    # io Collapsible button
    #
    ioCollapsibleButton = ctk.ctkCollapsibleButton()
    ioCollapsibleButton.text = "Volume and Transform Parameters"
    self.layout.addWidget(ioCollapsibleButton)

    # Layout within the parameter collapsible button
    ioFormLayout = qt.QFormLayout(ioCollapsibleButton)

    # Fixed Volume node selector
    self.fixedSelector = slicer.qMRMLNodeComboBox()
    self.fixedSelector.objectName = 'fixedSelector'
    self.fixedSelector.toolTip = "The fixed volume."
    self.fixedSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.fixedSelector.noneEnabled = False
    self.fixedSelector.addEnabled = False
    self.fixedSelector.removeEnabled = False
    ioFormLayout.addRow("Fixed Volume:", self.fixedSelector)
    self.fixedSelector.setMRMLScene(slicer.mrmlScene)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        self.fixedSelector, 'setMRMLScene(vtkMRMLScene*)')

    # Moving Volume node selector
    self.movingSelector = slicer.qMRMLNodeComboBox()
    self.movingSelector.objectName = 'movingSelector'
    self.movingSelector.toolTip = "The moving volume."
    self.movingSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.movingSelector.noneEnabled = False
    self.movingSelector.addEnabled = False
    self.movingSelector.removeEnabled = False
    ioFormLayout.addRow("Moving Volume:", self.movingSelector)
    self.movingSelector.setMRMLScene(slicer.mrmlScene)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        self.movingSelector, 'setMRMLScene(vtkMRMLScene*)')

    # Transform node selector
    self.transformSelector = slicer.qMRMLNodeComboBox()
    self.transformSelector.objectName = 'transformSelector'
    self.transformSelector.toolTip = "The transform volume."
    self.transformSelector.nodeTypes = ['vtkMRMLLinearTransformNode']
    self.transformSelector.noneEnabled = False
    self.transformSelector.addEnabled = False
    self.transformSelector.removeEnabled = False
    ioFormLayout.addRow("Moving To Fixed Transform:", self.transformSelector)
    self.transformSelector.setMRMLScene(slicer.mrmlScene)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        self.transformSelector, 'setMRMLScene(vtkMRMLScene*)')
    selectors = (self.fixedSelector, self.movingSelector, self.transformSelector)
    for selector in selectors:
      selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateLogicFromGUI)

    #
    # optimizer Collapsible button
    #
    optCollapsibleButton = ctk.ctkCollapsibleButton()
    optCollapsibleButton.text = "Optimizer Parameters"
    self.layout.addWidget(optCollapsibleButton)

    # Layout within the parameter collapsible button
    optFormLayout = qt.QFormLayout(optCollapsibleButton)

    # sample spacing
    # TODO: change this to be a pixel fraction (then scale spacings accordingly)
    self.sampleSpacingSlider = ctk.ctkSliderWidget()
    self.sampleSpacingSlider.decimals = 2
    self.sampleSpacingSlider.singleStep = 0.01
    self.sampleSpacingSlider.minimum = 0.5
    self.sampleSpacingSlider.maximum = 100
    self.sampleSpacingSlider.toolTip = "Multiple of spacing used when extracting pixels to evaluate objective function"
    optFormLayout.addRow("Sample Spacing:", self.sampleSpacingSlider)

    # gradient window slider
    self.gradientWindowSlider = ctk.ctkSliderWidget()
    self.gradientWindowSlider.decimals = 2
    self.gradientWindowSlider.singleStep = 0.01
    self.gradientWindowSlider.minimum = 0.01
    self.gradientWindowSlider.maximum = 5
    self.gradientWindowSlider.toolTip = "Multiple of spacing used when estimating objective function gradient"
    optFormLayout.addRow("Gradient Window:", self.gradientWindowSlider)

    # step size slider
    self.stepSizeSlider = ctk.ctkSliderWidget()
    self.stepSizeSlider.decimals = 2
    self.stepSizeSlider.singleStep = 0.01
    self.stepSizeSlider.minimum = 0.01
    self.stepSizeSlider.maximum = 5
    self.stepSizeSlider.toolTip = "Multiple of spacing used when taking an optimization step"
    optFormLayout.addRow("Step Size:", self.stepSizeSlider)

    # TODO: add interpolation options: nearest, linear, cubic 

    # TODO: add input image normaliztaion

    # TODO: add MI or other metrics

    # TODO: add output chart showing optimizer values

    # TODO: add quaternion optimization
    # -> scale for rotational gradient
    # OR: add multipoint localized gradient estimation and
    #     back-calculate rotation based on that



    # get default values from logic
    self.sampleSpacingSlider.value = self.logic.sampleSpacing
    self.gradientWindowSlider.value = self.logic.gradientWindow
    self.stepSizeSlider.value = self.logic.stepSize

    sliders = (self.sampleSpacingSlider, self.gradientWindowSlider, self.stepSizeSlider)
    for slider in sliders:
      slider.connect('valueChanged(double)', self.updateLogicFromGUI)

    # Run button
    self.runButton = qt.QPushButton("Run")
    self.runButton.toolTip = "Run registration bot."
    self.runButton.checkable = True
    optFormLayout.addRow(self.runButton)
    self.runButton.connect('toggled(bool)', self.onRunButtonToggled)

    # to support quicker development:
    import os
    if os.environ['USER'] == 'pieper':
      self.logic.testingData()
      self.fixedSelector.setCurrentNode(slicer.util.getNode('MRHead*'))
      self.movingSelector.setCurrentNode(slicer.util.getNode('neutral*'))
      self.transformSelector.setCurrentNode(slicer.util.getNode('movingToFixed*'))

    # Add vertical spacer
    self.layout.addStretch(1)

  def updateLogicFromGUI(self,args):
    self.logic.fixed = self.fixedSelector.currentNode()
    self.logic.moving = self.movingSelector.currentNode()
    self.logic.transform = self.transformSelector.currentNode()
    self.logic.sampleSpacing = self.sampleSpacingSlider.value
    self.logic.gradientWindow = self.gradientWindowSlider.value
    self.logic.stepSize = self.stepSizeSlider.value

  def onRunButtonToggled(self, checked):
    if checked:
      self.logic.start()
      self.runButton.text = "Stop"
    else:
      self.logic.stop()
      self.runButton.text = "Run"

  def onReload(self,moduleName="Regimatic"):
    """Generic reload method for any scripted module.
    ModuleWizard will subsitute correct default moduleName.
    """
    import imp, sys, os, slicer

    widgetName = moduleName + "Widget"

    # reload the source code
    # - set source file path
    # - load the module to the global space
    filePath = eval('slicer.modules.%s.path' % moduleName.lower())
    p = os.path.dirname(filePath)
    if not sys.path.__contains__(p):
      sys.path.insert(0,p)
    fp = open(filePath, "r")
    globals()[moduleName] = imp.load_module(
        moduleName, fp, filePath, ('.py', 'r', imp.PY_SOURCE))
    fp.close()

    # rebuild the widget
    # - find and hide the existing widget
    # - remove all the layout items
    # - create a new widget in the existing parent
    parent = slicer.util.findChildren(name='%s Reload' % moduleName)[0].parent()
    for child in parent.children():
      try:
        child.hide()
      except AttributeError:
        pass
    item = parent.layout().itemAt(0)
    while item:
      parent.layout().removeItem(item)
      item = parent.layout().itemAt(0)
    globals()[widgetName.lower()] = eval(
        'globals()["%s"].%s(parent)' % (moduleName, widgetName))
    globals()[widgetName.lower()].setup()

#
# Regimatic logic
#

class RegimaticLogic(object):
  """ Implement a template matching optimizer that is
  integrated with the slicer main loop.
  """

  def __init__(self,fixed=None,moving=None,transform=None):
    self.interval = 20
    self.timer = None

    # parameter defaults
    self.sampleSpacing = 10
    self.gradientWindow = 1
    self.stepSize = 1

    # slicer nodes set by the GUI
    self.fixed = fixed
    self.moving = moving
    self.transform = transform

    # optimizer state variables
    self.iteration = 0
    self.metric = 0

    # helper objects
    self.scratchMatrix = vtk.vtkMatrix4x4()
    self.ijkToRAS = vtk.vtkMatrix4x4()
    self.rasToIJK = vtk.vtkMatrix4x4()
    self.reslice = vtk.vtkImageReslice()
    self.resliceTransform = vtk.vtkTransform()
    self.viewer = None

  def start(self):
    """Create the subprocess and set up a polling timer"""
    if self.timer:
      self.stop()
    self.timer = qt.QTimer()
    self.timer.setInterval(self.interval)
    self.timer.connect('timeout()', self.tick)
    self.timer.start()

  def stop(self):
    if self.timer:
      self.timer.stop()
      self.timer = None

  def f(self,deltaM):
    """Evaluate the metric with an offset deltaM from the current transform"""
    movingRASArray = self.rasArray(self.moving, deltaM, self.fixed)
    pixelCount = reduce(lambda x,y: x*y, movingRASArray.shape)
    return numpy.sum(numpy.abs(movingRASArray-self.fixedRASArray)) / (1.0 * pixelCount)

  def dfdm(self):
    """Evaluate the gradient"""
    deltaM = vtk.vtkMatrix4x4()
    gradient = numpy.array((0.,0.,0.))
    oneOver2Window = 1. / (2*self.gradientWindow)
    for direction in xrange(3):
      deltaM.SetElement(direction,3,self.gradientWindow)
      mplus = self.f(deltaM)
      deltaM.SetElement(direction,3,-self.gradientWindow)
      mminus = self.f(deltaM)
      gradient[direction] = (mplus - mminus) * oneOver2Window
    return gradient

  def tick(self):
    """Callback for an iteration of the registration method
    """

    self.fixedRASArray = self.rasArray(self.fixed, None, self.fixed)
    movingRASArray = self.rasArray(self.moving, None, self.fixed)
    self.metric = numpy.sum(numpy.abs(movingRASArray-self.fixedRASArray))
    m = self.transform.GetMatrixTransformToParent()

    gradient = self.dfdm()

    self.transform.SetDisableModifiedEvent(True)
    for direction in xrange(3):
      oldValue = m.GetElement(direction,3)
      step = self.stepSize * gradient[direction]
      m.SetElement(direction,3,oldValue+step)
    self.transform.SetDisableModifiedEvent(False)
    self.transform.InvokePendingModifiedEvent()

    self.iteration += 1
    slicer.util.showStatusMessage("%s: %s (%s)" % (
      str(self.iteration),
      str(self.metric),
      str(gradient)))

  def rasArray(self, volumeNode, matrix=None, targetNode=None, debug=True):
    """
    Returns a numpy array of the given node resampled into RAS space
    If given, use the passed matrix as a final RAS to RAS transform
    """

    # get the transform from image space to world space
    volumeNode.GetIJKToRASMatrix(self.ijkToRAS)
    transformNode = volumeNode.GetParentTransformNode()
    if transformNode:
      self.scratchMatrix.Identity()
      transformNode.GetMatrixTransformToWorld(self.scratchMatrix)
      self.ijkToRAS.Multiply4x4(self.scratchMatrix, self.ijkToRAS, self.ijkToRAS)

    if matrix:
      self.ijkToRAS.Multiply4x4(matrix, self.ijkToRAS, self.ijkToRAS)

    self.rasToIJK.DeepCopy(self.ijkToRAS)
    self.rasToIJK.Invert()

    # use the matrix to extract the volume and convert it to an array

    self.reslice.SetInterpolationModeToLinear()
    self.reslice.InterpolateOn()
    self.resliceTransform.SetMatrix(self.rasToIJK)
    self.reslice.SetResliceTransform(self.resliceTransform)
    self.reslice.SetInput( volumeNode.GetImageData() )

    if targetNode:
      bounds = [0,]*6
      targetNode.GetRASBounds(bounds)
      self.reslice.SetOutputExtent(0, (bounds[1]-bounds[0])/self.sampleSpacing,
                                   0, (bounds[3]-bounds[2])/self.sampleSpacing,
                                   0, (bounds[5]-bounds[4])/self.sampleSpacing)
      self.reslice.SetOutputOrigin(bounds[0],bounds[2],bounds[4])
    self.reslice.SetOutputSpacing([self.sampleSpacing,]*3)

    self.reslice.UpdateWholeExtent()
    rasImage = self.reslice.GetOutput()
    shape = list(rasImage.GetDimensions())
    shape.reverse()
    rasArray = vtk.util.numpy_support.vtk_to_numpy(rasImage.GetPointData().GetScalars()).reshape(shape)


    if debug:
      if not self.viewer:
          self.viewer = vtk.vtkImageViewer()
      self.viewer.SetColorWindow( 500 )
      self.viewer.SetColorLevel( 200 )
      self.viewer.SetInput( rasImage )
      self.viewer.SetZSlice( rasArray.shape[2]/2 )
      self.viewer.Render()

    return rasArray

  def testingData(self):
    """Load some default data for development
    and set up a transform and viewing scenario for it.
    """
    if not slicer.util.getNodes('MRHead*'):
      import os
      fileName = os.environ['HOME'] + "/Dropbox/data/regmatic/MR-head.nrrd"
      vl = slicer.modules.volumes.logic()
      volumeNode = vl.AddArchetypeScalarVolume (fileName, "MRHead", 0, None)
    if not slicer.util.getNodes('neutral*'):
      import os
      fileName = os.environ['HOME'] + "/Dropbox/data/regmatic/neutral.nrrd"
      vl = slicer.modules.volumes.logic()
      volumeNode = vl.AddArchetypeScalarVolume (fileName, "neutral", 0, None)
    if not slicer.util.getNodes('movingToFixed'):
      # Create transform node
      transform = slicer.vtkMRMLLinearTransformNode()
      transform.SetName('movingToFixed')
      slicer.mrmlScene.AddNode(transform)
    head = slicer.util.getNode('MRHead')
    neutral = slicer.util.getNode('neutral')
    transform = slicer.util.getNode('movingToFixed')
    neutral.SetAndObserveTransformNodeID(transform.GetID())
    compositeNodes = slicer.util.getNodes('vtkMRMLSliceCompositeNode*')
    for compositeNode in compositeNodes.values():
      compositeNode.SetBackgroundVolumeID(head.GetID())
      compositeNode.SetForegroundVolumeID(neutral.GetID())
      compositeNode.SetForegroundOpacity(0.5)
    applicationLogic = slicer.app.applicationLogic()
    applicationLogic.FitSliceToAll()


  """rotation/quaternion utilities
    See:  http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Conversion_to_and_from_the_matrix_representation
  """

  def rotation(self,quat):
    """ build and return a 3x3 transformation matrix from a quaternion
        >>> rotation([1,0,0,0])
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])
    """

    quat = numpy.asmatrix(quat, dtype='Float64')
    quat /= numpy.sqrt(quat * quat.transpose())
    a,b,c,d           =  numpy.asarray(quat)[0]
    aa, bb, cc, dd    =  a*a, b*b, c*c, d*d
    ab, ac, ad        =  a*b, a*c, a*d
    bc, bd, cd        =  b*c, b*d, c*d

    Rot = numpy.matrix([[ aa + bb - cc - dd,  2.0 * (-ad + bc),    2.0*(ac + bd)     ],
                      [ 2.0 * (ad + bc),   aa - bb + cc - dd,  2.0 * (-ab + cd)   ],
                      [ 2.0 * (ac + bd),    2.0 * (ab + cd),   aa - bb - cc + dd ]])
    return(Rot)


  def quaternion(self,Rot):
    """ extract the quaternion from a matrix (assume upper 3x3 is rotation only)
    """

    # find the permutation that is most stable (largest diagonal first)
    xx, yy, zz = abs(Rot[0,0]), abs(Rot[1,1]), abs(Rot[2,2])
    if xx > yy and xx > zz:
      u, v, w = 0, 1, 2
    else:
      if yy > xx and yy > zz:
        u, v, w = 1, 2, 0
      else:
        u, v, w = 2, 0, 1

    # fill in the quat
    r = numpy.sqrt( 1 + Rot[u,u] - Rot[v,v] - Rot[w,w] )
    r2 = 2.*r

    if r < 1e-6:
      return numpy.array([1,0,0,0])

    quat = numpy.zeros(4)
    quat[  0] = (Rot[w,v] - Rot[v,w])/r2
    quat[1+u] = r/2.
    quat[1+v] = (Rot[u,v] + Rot[v,u])/r2
    quat[1+w] = (Rot[w,u] + Rot[u,w])/r2

    return quat

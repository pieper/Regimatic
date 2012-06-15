from __main__ import vtk, qt, ctk, slicer

#
# Regmatic
#

class Regmatic:
  def __init__(self, parent):
    parent.title = "Regmatic"
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
# qRegmaticWidget
#

class RegmaticWidget:
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

    self.logic = RegmaticLogic()

  def setup(self):
    # Instantiate and connect widgets ...

    # reload button
    self.reloadButton = qt.QPushButton("Reload")
    self.reloadButton.toolTip = "Reload this module."
    self.reloadButton.name = "Regmatic Reload"
    self.layout.addWidget(self.reloadButton)
    self.reloadButton.connect('clicked()', self.onReload)

    # Collapsible button
    parameterCollapsibleButton = ctk.ctkCollapsibleButton()
    parameterCollapsibleButton.text = "Regmatic Parameters"
    self.layout.addWidget(parameterCollapsibleButton)

    # Layout within the parameter collapsible button
    parameterFormLayout = qt.QFormLayout(parameterCollapsibleButton)

    # Fixed Volume node selector
    self.fixedSelector = slicer.qMRMLNodeComboBox()
    self.fixedSelector.objectName = 'fixedSelector'
    self.fixedSelector.toolTip = "The fixed volume."
    self.fixedSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.fixedSelector.noneEnabled = False
    self.fixedSelector.addEnabled = False
    self.fixedSelector.removeEnabled = False
    parameterFormLayout.addRow("Fixed Volume:", self.fixedSelector)
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
    parameterFormLayout.addRow("Moving Volume:", self.movingSelector)
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
    parameterFormLayout.addRow("Moving To Fixed Transform:", self.transformSelector)
    self.transformSelector.setMRMLScene(slicer.mrmlScene)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        self.transformSelector, 'setMRMLScene(vtkMRMLScene*)')
    selectors = (self.fixedSelector, self.movingSelector, self.transformSelector)
    for selector in selectors:
      selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateLogicFromGUI)

    # Run button
    self.runButton = qt.QPushButton("Run")
    self.runButton.toolTip = "Run registration bot."
    self.runButton.checkable = True
    parameterFormLayout.addRow(self.runButton)
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

  def updateLogicFromGUI(self):
    self.logic.fixed = self.fixedSelector.currentNode()
    self.logic.moving = self.movingSelector.currentNode()
    self.logic.transform = self.transformSelector.currentNode()

  def onRunButtonToggled(self, checked):
    if checked:
      self.logic.start()
      self.runButton.text = "Stop"
    else:
      self.logic.stop()
      self.runButton.text = "Run"

  def onReload(self,moduleName="Regmatic"):
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
    # - create a new widget in the existing parent
    parent = slicer.util.findChildren(name='%s Reload' % moduleName)[0].parent()
    for child in parent.children():
      try:
        child.hide()
      except AttributeError:
        pass
    globals()[widgetName.lower()] = eval(
        'globals()["%s"].%s(parent)' % (moduleName, widgetName))
    globals()[widgetName.lower()].setup()

#
# Regmatic logic
#

class RegmaticLogic(object):
  """ Implement a template matching optimizer that is
  integrated with the slicer main loop.
  Note: currently depends on numpy/scipy installation in mac system
  """

  def __init__(self,fixed=None,moving=None,transform=None):
    self.interval = 20
    self.timer = None

    # slicer nodes set by the GUI
    self.fixed = fixed
    self.moving = moving
    self.transform = transform

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

  def tick(self):
    """Callback for an iteration of the registration method
    """
    pass



  def testingData(self):
    """Load some default data for development
    and set up a transform and viewing scenario for it.
    """
    if not slicer.util.getNodes('MRHead*'):
      import os
      fileName = os.environ['HOME'] + "/Dropbox/data/regmatic/MR-head.nrrd"
      vl = slicer.modules.volumes.logic()
      volumeNode = vl.AddArchetypeScalarVolume (fileName, "MRHead", 0)
    if not slicer.util.getNodes('neutral*'):
      import os
      fileName = os.environ['HOME'] + "/Dropbox/data/regmatic/neutral.nrrd"
      vl = slicer.modules.volumes.logic()
      volumeNode = vl.AddArchetypeScalarVolume (fileName, "neutral", 0)
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

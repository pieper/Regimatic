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
    fixedSelector = slicer.qMRMLNodeComboBox()
    fixedSelector.objectName = 'fixedSelector'
    fixedSelector.toolTip = "The fixed volume."
    fixedSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    fixedSelector.noneEnabled = False
    fixedSelector.addEnabled = False
    fixedSelector.removeEnabled = False
    parameterFormLayout.addRow("Fixed Volume:", fixedSelector)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)', 
                        fixedSelector, 'setMRMLScene(vtkMRMLScene*)')

    # Moving Volume node selector
    movingSelector = slicer.qMRMLNodeComboBox()
    movingSelector.objectName = 'movingSelector'
    movingSelector.toolTip = "The moving volume."
    movingSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    movingSelector.noneEnabled = False
    movingSelector.addEnabled = False
    movingSelector.removeEnabled = False
    parameterFormLayout.addRow("Moving Volume:", movingSelector)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)', 
                        movingSelector, 'setMRMLScene(vtkMRMLScene*)')

    # Transform node selector
    transformSelector = slicer.qMRMLNodeComboBox()
    transformSelector.objectName = 'transformSelector'
    transformSelector.toolTip = "The transform volume."
    transformSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    transformSelector.noneEnabled = False
    transformSelector.addEnabled = False
    transformSelector.removeEnabled = False
    parameterFormLayout.addRow("Moving To Fixed Transform:", transformSelector)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)', 
                        transformSelector, 'setMRMLScene(vtkMRMLScene*)')

    # to support quicker development:
    import os
    if os.environ['USER'] == 'pieper':
      RegmaticLogic().testingData()

    # Add vertical spacer
    self.layout.addStretch(1)

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

  def __init__(self):
    pass

  def testingData(self):
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
    if not slicer.util.getNodes('Transform-head'):
      # Create transform node
      transform = slicer.vtkMRMLLinearTransformNode()
      transform.SetName('Transform-head')
      slicer.mrmlScene.AddNode(transform)
    head = slicer.util.getNode('MRHead')
    neutral = slicer.util.getNode('neutral')
    transform = slicer.util.getNode('Transform-head')
    neutral.SetAndObserveTransformNodeID(transform.GetID())
    compositeNodes = slicer.util.getNodes('vtkMRMLSliceCompositeNode*')
    for compositeNode in compositeNodes.values():
      compositeNode.SetBackgroundVolumeID(head.GetID())
      compositeNode.SetForegroundVolumeID(neutral.GetID())
      compositeNode.SetForegroundOpacity(0.5)
    applicationLogic = slicer.app.applicationLogic()
    applicationLogic.FitSliceToAll()

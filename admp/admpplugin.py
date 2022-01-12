# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-01-12
# version: 0.0.1

from admp.pme import ADMPPmeForce
import warnings

class ADMPPmeGenerator(object):

    #=============================================================================================

    """An ADMPPmeGenerator constructs ADMPPmeForce."""

    #=============================================================================================

    def __init__(self, forceField, scaleFactor14, defaultTholeWidth):
        self.forceField = forceField
        self.scaleFactor14 = scaleFactor14
        self.defaultTholeWidth = defaultTholeWidth
        self.typeMap = {}

    #=============================================================================================
    # Set axis type
    #=============================================================================================

    @staticmethod
    def setAxisType(kIndices):
        ZThenX = 0
        Bisector = 1
        ZBisect = 2
        ThreeFold = 3
        ZOnly = 4
        NoAxisType = 5
        LastAxisTypeIndex = 6    
        kStrings = ['kz', 'kx', 'ky']

        # set axis type

        kIndicesLen = len(kIndices)

        if (kIndicesLen > 3):
            ky = kIndices[3]
            kyNegative = False
            if ky.startswith('-'):
                ky = kIndices[3] = ky[1:]
                kyNegative = True
        else:
            ky = ""

        if (kIndicesLen > 2):
            kx = kIndices[2]
            kxNegative = False
            if kx.startswith('-'):
                kx = kIndices[2] = kx[1:]
                kxNegative = True
        else:
            kx = ""

        if (kIndicesLen > 1):
            kz = kIndices[1]
            kzNegative = False
            if kz.startswith('-'):
                kz = kIndices[1] = kz[1:]
                kzNegative = True
        else:
            kz = ""

        while(len(kIndices) < 4):
            kIndices.append("")

        axisType = ZThenX
        if (not kz):
            axisType = NoAxisType
        if (kz and not kx):
            axisType = ZOnly
        if (kz and kzNegative or kx and kxNegative):
            axisType = Bisector
        if (kx and kxNegative and ky and kyNegative):
            axisType = ZBisect
        if (kz and kzNegative and kx and kxNegative and ky and kyNegative):
            axisType = ThreeFold

        return axisType

    #=============================================================================================

    @staticmethod
    def parseElement(element, forceField):

        #   <ADMPPmeForce >
        # <Multipole class="1"    kz="2"    kx="4"    c0="-0.22620" d1="0.08214" d2="0.00000" d3="0.34883" q11="0.11775" q21="0.00000" q22="-1.02185" q31="-0.17555" q32="0.00000" q33="0.90410"  />
        # <Multipole class="2"    kz="1"    kx="3"    c0="-0.15245" d1="0.19517" d2="0.00000" d3="0.19687" q11="-0.20677" q21="0.00000" q22="-0.48084" q31="-0.01672" q32="0.00000" q33="0.68761"  />

        existing = [f for f in forceField._forces if isinstance(f, ADMPPmeGenerator)]
        if len(existing) == 0:
            generator = ADMPPmeGenerator(forceField, element.get('coulomb14scale', None), element.get('defaultTholeWidth', None))
            forceField.registerGenerator(generator)
        else:
            # Multiple <ADMPPmeForce> tags were found, probably in different files.  Simply add more types to the existing one.
            generator = existing[0]
            if abs(generator.scaleFactor14 != element.get('coulomb14scale', None)):
                raise ValueError('Found multiple ADMPPmeForce tags with different coulomb14scale arguments')
            if abs(generator.defaultTholeWidth != element.get('defaultTholeWidth', None)):
                raise ValueError('Found multiple ADMPPmeForce tags with different defaultTholeWidth arguments')

        # set type map: [ kIndices, multipoles, AMOEBA/OpenMM axis type]

        for atom in element.findall('Multipole'):
            types = forceField._findAtomTypes(atom.attrib, 1)
            if None not in types:

                # k-indices not provided default to 0

                kIndices = [atom.attrib['type']]

                kStrings = [ 'kz', 'kx', 'ky' ]
                for kString in kStrings:
                    try:
                        if (atom.attrib[kString]):
                             kIndices.append(atom.attrib[kString])
                    except:
                        pass

                # set axis type based on k-Indices

                axisType = ADMPPmeGenerator.setAxisType(kIndices)

                # set multipole

                charge = float(atom.get('c0'))

                conversion = 1.0
                dipole = [ conversion*float(atom.get('dX', 0.0)),
                           conversion*float(atom.get('dY', 0.0)),
                           conversion*float(atom.get('dZ', 0.0)) ]

                quadrupole = []
                quadrupole.append(conversion*float(atom.get('qXX', 0.0)))
                quadrupole.append(conversion*float(atom.get('qXY', 0.0)))
                quadrupole.append(conversion*float(atom.get('qYY', 0.0)))
                quadrupole.append(conversion*float(atom.get('qXZ', 0.0)))
                quadrupole.append(conversion*float(atom.get('qYZ', 0.0)))
                quadrupole.append(conversion*float(atom.get('qZZ', 0.0)))

                octopole = []
                octopole.append(conversion*float(atom.get('oXXX', 0.0)))
                octopole.append(conversion*float(atom.get('oXXY', 0.0)))
                octopole.append(conversion*float(atom.get('oXYY', 0.0)))
                octopole.append(conversion*float(atom.get('oYYY', 0.0)))
                octopole.append(conversion*float(atom.get('oXXZ', 0.0)))
                octopole.append(conversion*float(atom.get('oXYZ', 0.0)))
                octopole.append(conversion*float(atom.get('oYYZ', 0.0)))
                octopole.append(conversion*float(atom.get('oXZZ', 0.0)))
                octopole.append(conversion*float(atom.get('oYZZ', 0.0)))
                octopole.append(conversion*float(atom.get('oZZZ', 0.0)))

                for t in types[0]:
                    if (t not in generator.typeMap):
                        generator.typeMap[t] = []

                    valueMap = dict()
                    valueMap['classIndex'] = atom.attrib['type']
                    valueMap['kIndices'] = kIndices
                    valueMap['charge'] = charge
                    valueMap['dipole'] = dipole
                    valueMap['quadrupole'] = quadrupole
                    valueMap['octopole'] = octopole
                    valueMap['axisType'] = axisType
                    generator.typeMap[t].append(valueMap)

            else:
                outputString = "ADMPPmeGenerator: error getting type for multipole: %s" % (atom.attrib['class'])
                raise ValueError(outputString)

        # polarization parameters

        for atom in element.findall('Polarize'):
            types = forceField._findAtomTypes(atom.attrib, 1)
            if None not in types:

                classIndex = atom.attrib['type']
                polarizability = [ float(atom.attrib['polarizabilityXX']),
                                   float(atom.attrib['polarizabilityYY']),
                                   float(atom.attrib['polarizabilityZZ']) ]
                thole = float(atom.attrib['thole'])

                for t in types[0]:
                    if (t not in generator.typeMap):
                        outputString = "ADMPPmeGenerator: polarize type not present: %s" % (atom.attrib['type'])
                        raise ValueError(outputString)
                    else:
                        typeMapList = generator.typeMap[t]
                        hit = 0
                        for (ii, typeMap) in enumerate(typeMapList):

                            if (typeMap['classIndex'] == classIndex):
                                typeMap['polarizability'] = polarizability
                                typeMap['thole'] = thole
                                typeMapList[ii] = typeMap
                                hit = 1

                        if (hit == 0):
                            outputString = "ADMPPmeGenerator: error getting type for polarize: class index=%s not in multipole list?" % (atom.attrib['class'])
                            raise ValueError(outputString)

            else:
                outputString = "ADMPPmeGenerator: error getting type for polarize: %s" % (atom.attrib['class'])
                raise ValueError(outputString)

    #=============================================================================================

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):

        methodMap = {self.forcefield.NoCutoff:ADMPPmeForce.NoCutoff,
                     self.forcefield.PME:ADMPPmeForce.PME,
                     self.forcefield.LJPME:ADMPPmeForce.PME}

        force = ADMPPmeForce()
        sys.addForce(force)
        if (nonbondedMethod not in methodMap):
            raise ValueError( "ADMPPmeForce: input cutoff method not available." )
        else:
            force.setNonbondedMethod(methodMap[nonbondedMethod])
        force.setCutoffDistance(nonbondedCutoff)

        if ('ewaldErrorTolerance' in args):
            force.setEwaldErrorTolerance(float(args['ewaldErrorTolerance']))

        force.setPolarizationType(ADMPPmeForce.Extrapolated)
        if ('polarization' in args):
            polarizationType = args['polarization']
            if (polarizationType.lower() == 'direct'):
                force.setPolarizationType(ADMPPmeForce.Direct)
            elif (polarizationType.lower() == 'mutual'):
                force.setPolarizationType(ADMPPmeForce.Mutual)
            elif (polarizationType.lower() == 'extrapolated'):
                force.setPolarizationType(ADMPPmeForce.Extrapolated)
            else:
                raise ValueError( "ADMPPmeForce: invalide polarization type: " + polarizationType)

        argval = float(args['coulomb14scale']) if 'coulomb14scale' in args else None
        myval = float(self.scaleFactor14) if self.scaleFactor14 else None
        if argval is not None:
            if myval is not None:
                if myval != argval:
                     warnings.warn( "Conflicting coulomb14scale values found in forcefield file ({}) and createSystem args ({}).  "
                                    "Using the value from createSystem's arguments".format(myval, argval))
            force.set14ScaleFactor(argval)
        else:
            if myval is not None:
                force.set14ScaleFactor(myval)

        argval = float(args['defaultTholeWidth']) if 'defaultTholeWidth' in args else None
        myval = float(self.defaultTholeWidth) if self.defaultTholeWidth else None
        if argval is not None:
            if myval is not None:
                if myval != argval:
                     warnings.warn( "Conflicting defaultTholeWidth values found in forcefield file ({}) and createSystem args ({}).  "
                                    "Using the value from createSystem's arguments".format(myval, argval))
            force.setDefaultTholeWidth(argval)
        else:
            if myval is not None:
                force.setDefaultTholeWidth(myval)

        if ('aEwald' in args):
            force.setAEwald(float(args['aEwald']))

        if ('pmeGridDimensions' in args):
            force.setPmeGridDimensions(args['pmeGridDimensions'])

        if ('mutualInducedMaxIterations' in args):
            force.setMutualInducedMaxIterations(int(args['mutualInducedMaxIterations']))

        if ('mutualInducedTargetEpsilon' in args):
            force.setMutualInducedTargetEpsilon(float(args['mutualInducedTargetEpsilon']))

        # add particles to force
        # throw error if particle type not available

        # get 1-2, 1-3, 1-4, 1-5 bonded sets

        # 1-2

        bonded12ParticleSets = self.forcefield.AmoebaVdwGenerator.getBondedParticleSets(sys, data)

        # 1-3

        bonded13ParticleSets = []
        for i in range(len(data.atoms)):
            bonded13Set = set()
            bonded12ParticleSet = bonded12ParticleSets[i]
            for j in bonded12ParticleSet:
                bonded13Set = bonded13Set.union(bonded12ParticleSets[j])

            # remove 1-2 and self from set

            bonded13Set = bonded13Set - bonded12ParticleSet
            selfSet = set()
            selfSet.add(i)
            bonded13Set = bonded13Set - selfSet
            bonded13Set = set(sorted(bonded13Set))
            bonded13ParticleSets.append(bonded13Set)

        # 1-4

        bonded14ParticleSets = []
        for i in range(len(data.atoms)):
            bonded14Set = set()
            bonded13ParticleSet = bonded13ParticleSets[i]
            for j in bonded13ParticleSet:
                bonded14Set = bonded14Set.union(bonded12ParticleSets[j])

            # remove 1-3, 1-2 and self from set

            bonded14Set = bonded14Set - bonded12ParticleSets[i]
            bonded14Set = bonded14Set - bonded13ParticleSet
            selfSet = set()
            selfSet.add(i)
            bonded14Set = bonded14Set - selfSet
            bonded14Set = set(sorted(bonded14Set))
            bonded14ParticleSets.append(bonded14Set)


        for (atomIndex, atom) in enumerate(data.atoms):
            t = data.atomType[atom]
            if t in self.typeMap:

                multipoleList = self.typeMap[t]
                hit = 0
                savedMultipoleDict = 0

                # assign multipole parameters via only 1-2 connected atoms

                for multipoleDict in multipoleList:

                    if (hit != 0):
                        break

                    kIndices = multipoleDict['kIndices']

                    kz = kIndices[1]
                    kx = kIndices[2]
                    ky = kIndices[3]

                    # assign multipole parameters
                    #    (1) get bonded partners
                    #    (2) match parameter types

                    bondedAtomIndices = bonded12ParticleSets[atomIndex]
                    zaxis = -1
                    xaxis = -1
                    yaxis = -1
                    for bondedAtomZIndex in bondedAtomIndices:

                       if (hit != 0):
                           break

                       bondedAtomZType = data.atomType[data.atoms[bondedAtomZIndex]]
                       bondedAtomZ = data.atoms[bondedAtomZIndex]
                       if (bondedAtomZType == kz):
                          for bondedAtomXIndex in bondedAtomIndices:
                              if (bondedAtomXIndex == bondedAtomZIndex or hit != 0):
                                  continue
                              bondedAtomXType = data.atomType[data.atoms[bondedAtomXIndex]]
                              if (bondedAtomXType == kx):
                                  if (not ky):
                                      zaxis = bondedAtomZIndex
                                      xaxis = bondedAtomXIndex
                                      if( bondedAtomXType == bondedAtomZType and xaxis < zaxis ):
                                          swapI = zaxis
                                          zaxis = xaxis
                                          xaxis = swapI
                                      else:
                                          for bondedAtomXIndex in bondedAtomIndices:
                                              bondedAtomX1Type = data.atomType[data.atoms[bondedAtomXIndex]]
                                              if( bondedAtomX1Type == kx and bondedAtomXIndex != bondedAtomZIndex and bondedAtomXIndex < xaxis ):
                                                  xaxis = bondedAtomXIndex

                                      savedMultipoleDict = multipoleDict
                                      hit = 1
                                  else:
                                      for bondedAtomYIndex in bondedAtomIndices:
                                          if (bondedAtomYIndex == bondedAtomZIndex or bondedAtomYIndex == bondedAtomXIndex or hit != 0):
                                              continue
                                          bondedAtomYType = data.atomType[data.atoms[bondedAtomYIndex]]
                                          if (bondedAtomYType == ky):
                                              zaxis = bondedAtomZIndex
                                              xaxis = bondedAtomXIndex
                                              yaxis = bondedAtomYIndex
                                              savedMultipoleDict = multipoleDict
                                              hit = 2

                # assign multipole parameters via 1-2 and 1-3 connected atoms

                for multipoleDict in multipoleList:

                    if (hit != 0):
                        break

                    kIndices = multipoleDict['kIndices']

                    kz = kIndices[1]
                    kx = kIndices[2]
                    ky = kIndices[3]

                    # assign multipole parameters
                    #    (1) get bonded partners
                    #    (2) match parameter types

                    bondedAtom12Indices = bonded12ParticleSets[atomIndex]
                    bondedAtom13Indices = bonded13ParticleSets[atomIndex]

                    zaxis = -1
                    xaxis = -1
                    yaxis = -1

                    for bondedAtomZIndex in bondedAtom12Indices:

                       if (hit != 0):
                           break

                       bondedAtomZType = data.atomType[data.atoms[bondedAtomZIndex]]
                       bondedAtomZ = data.atoms[bondedAtomZIndex]

                       if (bondedAtomZType == kz):
                          for bondedAtomXIndex in bondedAtom13Indices:

                              if (bondedAtomXIndex == bondedAtomZIndex or hit != 0):
                                  continue
                              bondedAtomXType = data.atomType[data.atoms[bondedAtomXIndex]]
                              if (bondedAtomXType == kx and bondedAtomZIndex in bonded12ParticleSets[bondedAtomXIndex]):
                                  if (not ky):
                                      zaxis = bondedAtomZIndex
                                      xaxis = bondedAtomXIndex

                                      # select xaxis w/ smallest index

                                      for bondedAtomXIndex in bondedAtom13Indices:
                                          bondedAtomX1Type = data.atomType[data.atoms[bondedAtomXIndex]]
                                          if( bondedAtomX1Type == kx and bondedAtomXIndex != bondedAtomZIndex and bondedAtomZIndex in bonded12ParticleSets[bondedAtomXIndex] and bondedAtomXIndex < xaxis ):
                                              xaxis = bondedAtomXIndex

                                      savedMultipoleDict = multipoleDict
                                      hit = 3
                                  else:
                                      for bondedAtomYIndex in bondedAtom13Indices:
                                          if (bondedAtomYIndex == bondedAtomZIndex or bondedAtomYIndex == bondedAtomXIndex or hit != 0):
                                              continue
                                          bondedAtomYType = data.atomType[data.atoms[bondedAtomYIndex]]
                                          if (bondedAtomYType == ky and bondedAtomZIndex in bonded12ParticleSets[bondedAtomYIndex]):
                                              zaxis = bondedAtomZIndex
                                              xaxis = bondedAtomXIndex
                                              yaxis = bondedAtomYIndex
                                              savedMultipoleDict = multipoleDict
                                              hit = 4

                # assign multipole parameters via only a z-defining atom

                for multipoleDict in multipoleList:

                    if (hit != 0):
                        break

                    kIndices = multipoleDict['kIndices']

                    kz = kIndices[1]
                    kx = kIndices[2]

                    zaxis = -1
                    xaxis = -1
                    yaxis = -1

                    for bondedAtomZIndex in bondedAtom12Indices:

                        if (hit != 0):
                            break

                        bondedAtomZType = data.atomType[data.atoms[bondedAtomZIndex]]
                        bondedAtomZ = data.atoms[bondedAtomZIndex]

                        if (not kx and kz == bondedAtomZType):
                            zaxis = bondedAtomZIndex
                            savedMultipoleDict = multipoleDict
                            hit = 5

                # assign multipole parameters via no connected atoms

                for multipoleDict in multipoleList:

                    if (hit != 0):
                        break

                    kIndices = multipoleDict['kIndices']

                    kz = kIndices[1]

                    zaxis = -1
                    xaxis = -1
                    yaxis = -1

                    if (not kz):
                        savedMultipoleDict = multipoleDict
                        hit = 6

                # add particle if there was a hit

                if (hit != 0):

                    atom.multipoleDict = savedMultipoleDict
                    atom.polarizationGroups = dict()
                    try:
                        thole = savedMultipoleDict['thole']
                    except KeyError:
                        thole = 0.0
                    try:
                        polarizability = savedMultipoleDict['polarizability']
                    except KeyError:
                        polarizability = [0.0, 0.0, 0.0]

                    newIndex = force.addMultipole(savedMultipoleDict['charge'], savedMultipoleDict['dipole'],
                                                  savedMultipoleDict['quadrupole'], savedMultipoleDict['octopole'], savedMultipoleDict['axisType'],
                                                  zaxis, xaxis, yaxis, thole, polarizability)
                    if (atomIndex == newIndex):
                        force.setCovalentMap(atomIndex, ADMPPmeForce.Covalent12, tuple(bonded12ParticleSets[atomIndex]))
                        force.setCovalentMap(atomIndex, ADMPPmeForce.Covalent13, tuple(bonded13ParticleSets[atomIndex]))
                        force.setCovalentMap(atomIndex, ADMPPmeForce.Covalent14, tuple(bonded14ParticleSets[atomIndex]))
                    else:
                        raise ValueError("Atom %s of %s %d is out of sync!." %(atom.name, atom.residue.name, atom.residue.index))
                else:
                    raise ValueError("Atom %s of %s %d was not assigned." %(atom.name, atom.residue.name, atom.residue.index))
            else:
                raise ValueError('No multipole type for atom %s %s %d' % (atom.name, atom.residue.name, atom.residue.index))

        self.forcefield.parsers["ADMPPmeForce"] = ADMPPmeGenerator.parseElement
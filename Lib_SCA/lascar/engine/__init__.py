# This file is part of lascar
#
# lascar is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#
# Copyright 2018 Manuel San Pedro, Victor Servant, Charles Guillemet, Ledger SAS - manuel.sanpedro@ledger.fr,
# victor.servant@ledger.fr, charles@ledger.fr

from .engine import ContainerDumpEngine
from .engine import MeanEngine
from .engine import VarEngine
from .engine import Engine
from .engine import PearsonCorrelationEngine
from .engine import GroupedEngines

from .guess_engine import GuessEngine
from .partitioner_engine import PartitionerEngine
from .cpa_engine import CpaEngine
from .cpa_engine import CpaPartitionedEngine
from .dpa_engine import DpaEngine
from .nicv_engine import NicvEngine
from .snr_engine import SnrEngine
from .ttest_engine import TTestEngine
from .ttest_engine import compute_ttest

from .chi2test_engine import Chi2TestEngine

from .classifier_engine import MatchEngine
from .classifier_engine import ProfileEngine
from .classifier_engine import save_classifier
from .classifier_engine import load_classifier

from .dom_engine import DomEngine
from .mi_engine import CMI_Engine_By_Histogram
from .mi_engine import CMI_Engine_By_KDE
from .success_rate_engine import numerical_success_rate

# from .graph_construction_basic import PhaseSpaceReconstructionGraph, AmplitudesBasedGraph, GeneralisedRDPG

from .graph_construction_test_engine import T_GraphConstructionPSR, T_GraphConstructionAB, T_GraphConstructionTraceBatch
from .graph_construction_test_engine import T_GraphConstructionTraceAllCorr, T_GraphConstructionTraceAllDist
from .graph_construction_attack_engine import A_GraphConstructionPSR, A_GraphConstructionAB, A_GraphConstructionTraceBatch
from .graph_construction_attack_engine import A_GraphConstructionTraceAllCorr_SB, A_GraphConstructionTraceAllCorr_MB, \
    A_GraphConstructionTraceAllHistogram_SB, A_GraphConstructionTraceAllHistogram_MB

from .graph_testing_engine import GraphTestEnginePSR, GraphTestEngineAB, GraphTestEngineTraceBatch
from .graph_testing_engine import GraphTestingEngineTraceAllCorr, GraphTestingEngineTraceAllDist
from .graph_attack_engine import GraphAttackEngineTraceAllCorr_SB, GraphAttackEngineTraceAllCorr_MB, \
    GraphAttackEngineTraceAllKSDist_SB, GraphAttackEngineTraceAllChi2_SB, GraphAttackEngineTraceAllChi2_MB

# from .graph_testing_basic import TwoSamplesGraphTesting, GraphCommunityBasedTesting, Chi2Test
# from .graph_attack_basic import SpectralDistance

from .graph_engine_mi_test import GraphMIEngine

# from .template_engine import TemplateProfileEngine
# from .template_engine import TemplateMatchEngine

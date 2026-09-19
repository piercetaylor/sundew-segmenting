"""Conservative growth-form priors for the taxa in the curated dataset.

These labels support sampling and error analysis. They are not segmentation
classes, and ``dense_mat`` is only a likely image presentation for pygmy taxa.
"""

from __future__ import annotations


GROWTH_FORMS = ("rosette", "erect_or_branching", "linear_or_forked", "dense_mat")

TAXON_GROWTH_FORMS = {
    "Drosera × obovata": "rosette",
    "Drosera � obovata": "rosette",  # legacy replacement character in one local export
    "Drosera aberrans": "rosette",
    "Drosera admirabilis": "rosette",
    "Drosera anglica": "rosette",
    "Drosera arcturi": "linear_or_forked",
    "Drosera auriculata": "erect_or_branching",
    "Drosera australis": "dense_mat",
    "Drosera binata": "linear_or_forked",
    "Drosera brevifolia": "rosette",
    "Drosera bulbosa": "rosette",
    "Drosera capensis": "linear_or_forked",
    "Drosera capillaris": "rosette",
    "Drosera cistiflora": "erect_or_branching",
    "Drosera collina": "rosette",
    "Drosera ericgreenii": "rosette",
    "Drosera erythrorhiza": "rosette",
    "Drosera filiformis": "linear_or_forked",
    "Drosera floridana": "linear_or_forked",
    "Drosera glabripes": "erect_or_branching",
    "Drosera glanduligera": "rosette",
    "Drosera grantsaui": "erect_or_branching",
    "Drosera graomogolensis": "erect_or_branching",
    "Drosera hilaris": "erect_or_branching",
    "Drosera hookeri": "erect_or_branching",
    "Drosera hyperostigma": "dense_mat",
    "Drosera intermedia": "erect_or_branching",
    "Drosera latifolia": "rosette",
    "Drosera linearis": "linear_or_forked",
    "Drosera madagascariensis": "erect_or_branching",
    "Drosera magna": "rosette",
    "Drosera minutiflora": "dense_mat",
    "Drosera murfetii": "linear_or_forked",
    "Drosera natalensis": "rosette",
    "Drosera nitidula": "dense_mat",
    "Drosera pallida": "erect_or_branching",
    "Drosera pauciflora": "rosette",
    "Drosera porrecta": "erect_or_branching",
    "Drosera pulchella": "dense_mat",
    "Drosera pygmaea": "dense_mat",
    "Drosera roraimae": "rosette",
    "Drosera roseana": "dense_mat",
    "Drosera rosulata": "rosette",
    "Drosera rotundifolia": "rosette",
    "Drosera sessilifolia": "rosette",
    "Drosera spatulata": "rosette",
    "Drosera spatulata gympiensis": "rosette",
    "Drosera spatulata spatulata": "rosette",
    "Drosera spilos": "dense_mat",
    "Drosera stenopetala": "rosette",
    "Drosera tomentosa": "rosette",
    "Drosera tracyi": "linear_or_forked",
    "Drosera trinervia": "rosette",
    "Drosera variegata": "erect_or_branching",
    "Drosera whittakeri": "rosette",
    "Drosera xerophila": "rosette",
    "Drosera zonaria": "rosette",
}


def growth_form_for_taxon(taxon_name: str) -> str:
    """Return a conservative taxon-level morphology prior."""
    return TAXON_GROWTH_FORMS.get(taxon_name, "unknown")

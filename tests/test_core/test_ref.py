import pytest
from pydantic import BaseModel, TypeAdapter

from conductor.ref import Ref


def test_a_ref_names_a_node_and_one_of_its_fields():
    ref = Ref("n1", "result")

    assert ref.node_id == "n1"
    assert ref.field == "result"


def test_a_ref_is_the_address_rather_than_a_rendering_of_it():
    """It goes where a key goes, as it is: there is no encode step to
    remember and no second spelling for a frontend to agree with."""
    assert Ref("n1", "result") == "n1.result"
    assert {Ref("n1", "result"): 1}["n1.result"] == 1


def test_composing_an_address_and_reading_one_give_the_same_value():
    """Two arities, two questions — an author has the parts, a reader has
    the address. Neither converts: the value is the same either way."""
    assert Ref("n1.result") == Ref("n1", "result")
    assert Ref(Ref("n1", "result")) == Ref("n1", "result")


def test_what_is_not_an_address_is_refused():
    """A string with no dot, or with an empty half, is not guessed at."""
    with pytest.raises(ValueError, match="address"):
        Ref("result")
    with pytest.raises(ValueError, match="address"):
        Ref("n1.")
    with pytest.raises(ValueError, match="placement"):
        Ref("a.b", "c")


def test_an_address_has_two_parts_because_depth_is_a_nodes_job():
    """A nested value arrives whole, on one field; the node that opens it
    births a child index and exposes one field per part. So there is
    nothing below a field to address and no path to carry — and the field
    is simply whatever follows the first dot."""
    assert Ref("n1", "Pris i kr.").field == "Pris i kr."
    assert Ref("emb", "sag.value").node_id == "emb"
    assert Ref("emb", "sag.value").field == "sag.value"
    assert Ref("emb.sag.value") == Ref("emb", Ref("sag", "value"))


def test_pydantic_keeps_the_type_and_dumps_the_address():
    """Without a core schema a ``str`` subclass comes back a bare ``str``
    and every ``ref.node_id`` downstream is an ``AttributeError`` — the
    same help a ``DType`` built on a builtin needs."""

    class Held(BaseModel):
        ref: Ref

    assert isinstance(Held(ref="n1.result").ref, Ref)
    assert Held(ref="n1.result").model_dump() == {"ref": "n1.result"}
    with pytest.raises(ValueError, match="address"):
        Held(ref="result")


def test_a_ref_keyed_map_is_a_json_object():
    """Which is the whole reason an address is a string: ``carried``,
    ``layout`` and a caller's answers are all keyed by one."""
    dumped = TypeAdapter(dict[Ref, int]).dump_python({Ref("a", "b"): 1}, mode="json")

    assert dumped == {"a.b": 1}

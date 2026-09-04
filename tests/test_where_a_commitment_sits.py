"""The storage key a reader uses to check a submission without asking us.

``storage_key`` is the whole of the subnet's "don't trust us, look" claim: it
addresses the exact storage item a commitment lives in, and a node's raw query
returns the same bytes the field was built from. A wrong key is worse than no
key - it addresses a real, empty slot, which reads exactly like a submission
that was never made.

Pinned against a value read out of polkadot.js rather than against the spec.
The endianness is the part that is easy to get wrong, and easy to get wrong
consistently in a second implementation.
"""

from __future__ import annotations

import re

import pytest

from capability_subnet.common.chain import AddressError, explorer_url, public_key, storage_key

ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
READ_FROM_THE_EXPLORER = (
    "0xca407206ec1ab726b2636c4b145ac287"
    "9055a2cffd4b296f38f2de8d9b333cab"
    "6700"
    "af89655d282089246a6b8d65deed2e3a430820461bca6b104708150b22f8cdfc159753e48364f36b"
)


class TestTheKeyIsTheOneTheChainUses:
    def test_it_reproduces_a_key_copied_from_the_explorer(self):
        assert (
            storage_key(
                "5EUEt9WW7j8E2M7Hvgeai8rMrdCZX6aMeBRtPkcgM9Mkkkkk", netuid=103, revealed=True
            )
            == READ_FROM_THE_EXPLORER
        )

    def test_the_netuid_is_readable_in_the_middle(self):
        """It sits under the identity hasher, so it is its own little-endian
        encoding rather than a hash. 103 -> 6700."""
        key = storage_key(ALICE, netuid=103, revealed=True)

        assert key[2 + 64 : 2 + 68] == "6700"

    def test_the_account_is_the_tail(self):
        key = storage_key(ALICE, netuid=103, revealed=True)

        assert key.endswith(public_key(ALICE).hex())

    def test_sealed_and_revealed_are_different_maps(self):
        """One key for both sends a reader looking for a sealed commitment into
        the revealed map, and back."""
        assert storage_key(ALICE, netuid=103, revealed=True) != storage_key(
            ALICE, netuid=103, revealed=False
        )

    def test_another_subnet_is_another_key(self):
        assert storage_key(ALICE, netuid=103, revealed=True) != storage_key(
            ALICE, netuid=1, revealed=True
        )


class TestItRunsWhereItIsCalled:
    def test_it_imports_no_substrate_codec_library(self):
        """The engine, its CI and any miner on the current stack carry none of
        these: bittensor 11 does not pull them in. An import of one here raises
        ModuleNotFoundError for every caller, and only at the call.

        Asserted on the import lines, so the reason can be written down in
        prose without tripping the check.
        """
        import pathlib

        from capability_subnet.common import chain

        forbidden = ("scalecodec", "substrateinterface", "bt_decode")
        offenders = [
            line
            for line in pathlib.Path(chain.__file__).read_text().splitlines()
            if re.match(r"\s*(import|from)\s+(" + "|".join(forbidden) + r")\b", line)
        ]
        assert not offenders, f"chain.py must decode addresses itself: {offenders}"


class TestAnAddressItCannotTrust:
    def test_a_mistyped_address_is_refused_not_decoded(self):
        """It decodes into 32 bytes that are not anybody's key. Without the
        checksum the caller gets a key to an empty slot instead of an error."""
        with pytest.raises(AddressError):
            public_key("5H6R2CaCVNonRRgfdYsfzEpe2ew88un42KpYoZsDL5oLa5jX")

    def test_something_that_is_not_an_address_at_all(self):
        with pytest.raises(AddressError):
            public_key("not-an-address")

    def test_an_empty_string(self):
        with pytest.raises(AddressError):
            public_key("")


class TestTheExplorerLink:
    def test_it_points_at_this_network(self):
        assert "entrypoint-finney" in explorer_url()

    def test_it_lands_on_the_chain_state_page(self):
        """Only as far as the page: polkadot.js takes the endpoint in the URL
        but not a query with its arguments, so the key does the rest."""
        assert explorer_url().endswith("#/chainstate")

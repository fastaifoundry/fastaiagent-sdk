"""The e2e harness signs in at a plane's login address (1.94.0; plane F-Q40 = A).

Against real planes, no stand-ins:

* a plane with no login proxy (``E2E_PLANE_*``, the usual local plane) — the
  lookup answers 404 and the plain sign-in still works;
* a plane behind the login proxy (``S7_PROXY_URL``), where a password sign-in
  without a ticket answers ``400 customer_required`` — the reason for the change
  — and the ticket flow signs in, choosing a workspace by name when the person
  is in several (``S7_CHOICE_EMAIL`` / ``S7_CHOICE_PASSWORD``, a person in two).

Each part skips when its plane is not configured.
"""

from __future__ import annotations

import os

import httpx
import pytest

from tests.e2e.conftest import plane_login

pytestmark = pytest.mark.e2e


def _need(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"not configured: {missing}")


def _domains(address: str, token: str) -> int:
    r = httpx.get(
        f"{address}/api/v1/users/me/domains",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    return r.status_code


def test_a_plane_without_a_proxy_still_signs_in() -> None:
    _need("FASTAIAGENT_TARGET", "E2E_PLANE_EMAIL", "E2E_PLANE_PASSWORD")
    target = os.environ["FASTAIAGENT_TARGET"]
    lookup = httpx.post(f"{target}/proxy/v1/login/lookup", json={"email": "x@y.zz"}, timeout=30)
    assert lookup.status_code in (404, 405), "this part needs a plane with no proxy"
    token = plane_login(target, os.environ["E2E_PLANE_EMAIL"], os.environ["E2E_PLANE_PASSWORD"])
    assert _domains(target, token) == 200


def test_the_login_address_refuses_a_sign_in_without_a_ticket() -> None:
    """What the harness did before 1.94.0, at a login address."""
    _need("S7_PROXY_URL", "S7_CHOICE_EMAIL", "S7_CHOICE_PASSWORD")
    r = httpx.post(
        f"{os.environ['S7_PROXY_URL']}/api/v1/auth/login",
        json={"email": os.environ["S7_CHOICE_EMAIL"], "password": os.environ["S7_CHOICE_PASSWORD"]},
        timeout=30,
    )
    assert r.status_code == 400 and r.json()["detail"]["error"] == "customer_required"


def test_a_person_in_two_workspaces_signs_in_to_the_named_one() -> None:
    _need("S7_PROXY_URL", "S7_CHOICE_EMAIL", "S7_CHOICE_PASSWORD")
    address = os.environ["S7_PROXY_URL"]
    email, password = os.environ["S7_CHOICE_EMAIL"], os.environ["S7_CHOICE_PASSWORD"]
    token = plane_login(address, email, password, workspace="Workspace 2")
    assert _domains(address, token) == 200


def test_a_choice_without_a_workspace_name_fails_and_names_them() -> None:
    _need("S7_PROXY_URL", "S7_CHOICE_EMAIL", "S7_CHOICE_PASSWORD")
    with pytest.raises(pytest.fail.Exception, match="Workspace 1.*Workspace 2"):
        plane_login(
            os.environ["S7_PROXY_URL"],
            os.environ["S7_CHOICE_EMAIL"],
            os.environ["S7_CHOICE_PASSWORD"],
        )


def test_a_wrong_password_at_the_login_address_fails() -> None:
    _need("S7_PROXY_URL", "S7_CHOICE_EMAIL")
    with pytest.raises(pytest.fail.Exception, match="HTTP 401"):
        plane_login(os.environ["S7_PROXY_URL"], os.environ["S7_CHOICE_EMAIL"], "Wrong-pass-1!")

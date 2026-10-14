"""Numerical and ingestion regressions; fixtures are not scientific results."""
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from data.astronomy_manifest import load_astronomy_manifest
from data.loaders.tng_loader import TNGLoader
from model.physics_loss import PhysicsInformedLoss, VirialLoss
from training.scheduler import LambdaScheduler
from training.train import Trainer
from tests.test_astronomy_manifest import _catalog, _write_manifest


def _physics_batch():
    return SimpleNamespace(stellar_mass=torch.tensor([1e10]),
                           vel_disp=torch.tensor([200.]),
                           half_mass_r=torch.tensor([.01]), batch=torch.tensor([0]))


def test_zero_lambda_bypasses_physics_without_required_observations():
    predictions = torch.tensor([0.], requires_grad=True)
    loss, components = PhysicsInformedLoss({})(
        predictions, torch.tensor([12.]), object(), lambda_weight=0.)
    assert loss.item() == 144.
    assert components['virial_loss'].item() == 0.
    loss.backward()
    assert predictions.grad.item() == -24.


@pytest.mark.parametrize('prediction', [0., -100., 100., -1000., 1000.])
def test_virial_extreme_predictions_have_finite_loss_diagnostics_and_gradients(prediction):
    predictions = torch.tensor([prediction], requires_grad=True)
    loss, components = VirialLoss({})(predictions, _physics_batch(), lambda_weight=.05)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in components.values())
    loss.backward()
    assert torch.isfinite(predictions.grad).all()


def test_zero_lambda_overflow_reproducer_returns_exact_zero():
    loss, _ = VirialLoss({})(torch.tensor([0.]), _physics_batch(), lambda_weight=0.)
    assert loss.item() == 0.


def test_zero_lambda_bypass_does_not_overflow_when_predictions_sum_overflows():
    predictions = torch.tensor([3e38, 3e38], requires_grad=True)
    loss, _ = VirialLoss({})(predictions, object(), lambda_weight=0.)
    assert loss.item() == 0.
    loss.backward()
    assert torch.equal(predictions.grad, torch.zeros(2))


def test_representable_virial_penalty_preserves_squared_ratio_objective():
    predictions = torch.tensor([10.], requires_grad=True)
    loss, components = VirialLoss({})(predictions, _physics_batch())
    # 2 KE=4e14, PE=4.302e13, in the documented Mpc/km/s/Msun units.
    ratio = 4e14 / 4.302e13
    assert components['virial_ratio_mean'].item() == pytest.approx(ratio, rel=1e-5)
    assert loss.item() == pytest.approx((ratio - 1.) ** 2, rel=1e-5)
    loss.backward()
    assert predictions.grad.item() < 0.


@pytest.mark.parametrize('weight', [-.1, float('nan'), float('inf')])
def test_virial_rejects_invalid_lambda_even_when_disabled(weight):
    with pytest.raises(ValueError, match='lambda'):
        VirialLoss({'physics': {'use_virial_loss': False}})(torch.tensor([12.]), object(), weight)


@pytest.mark.parametrize('key,value', [('G', 0.), ('G', float('nan')),
                                      ('virial_coefficient', -1.),
                                      ('virial_log_ratio_bound', float('inf'))])
def test_virial_rejects_invalid_numerical_configuration(key, value):
    with pytest.raises(ValueError, match=key):
        VirialLoss({'physics': {key: value}})


def test_cosine_scheduler_honors_zero_start_instead_of_hidden_floor():
    scheduler = LambdaScheduler({'training': {'lambda_schedule': 'cosine',
        'lambda_start': 0., 'lambda_end': .05, 'lambda_epochs': 4}})
    assert scheduler.step(0) == 0.
    assert scheduler.step(2) == pytest.approx(.025)
    assert scheduler.step(4) == .05


@pytest.mark.parametrize('schedule,wanted', [('linear', .03), ('cosine', .03), ('fixed', .07)])
def test_trainer_uses_training_lambda_configuration(tmp_path, schedule, wanted):
    config = {'training': {'epochs': 20, 'lambda_schedule': schedule,
        'lambda_start': .01, 'lambda_end': .05, 'lambda_epochs': 4,
        'lambda_fixed': .07, 'lr_scheduler': 'none',
        'checkpoint_dir': str(tmp_path)}, 'wandb': {'enabled': False},
        'lambda_end': .9, 'warmup_epochs': 75}
    config['training']['warmup_epochs'] = 0
    trainer = Trainer(config, torch.nn.Linear(1, 1), [], [], device=torch.device('cpu'))
    assert trainer.lambda_scheduler.step(2) == pytest.approx(wanted)


def test_lambda_warmup_delays_configured_ramp():
    scheduler = LambdaScheduler({'training': {'warmup_epochs': 2,
        'lambda_schedule': 'linear', 'lambda_start': .01,
        'lambda_end': .05, 'lambda_epochs': 4}})
    assert [scheduler.step(i) for i in (0, 1, 2, 4, 6)] == pytest.approx([0., 0., .01, .03, .05])


def test_explicit_legacy_root_scheduler_values_remain_supported():
    scheduler = LambdaScheduler({'warmup_epochs': 2, 'lambda_end': .005,
        'training': {'epochs': 10, 'lambda_start': 0.}})
    assert scheduler.step(0) == 0.
    assert scheduler.step(6) == pytest.approx(.0025)
    assert scheduler.step(10) == .005


@pytest.mark.parametrize('settings', [{'lambda_schedule': 'typo'}, {'lambda_end': -1.},
    {'warmup_epochs': -1}, {'lambda_epochs': 0}, {'lambda_start': float('nan')}])
def test_scheduler_rejects_invalid_configuration(settings):
    with pytest.raises(ValueError):
        LambdaScheduler({'training': settings})


def _tng_loader(tmp_path, **settings):
    return TNGLoader({'data': {'tng': {'cache_dir': str(tmp_path / 'cache'),
        'min_subhalos_per_halo': 1, **settings}}})


def _api_subhalo():
    return {'id': 8, 'SubhaloPos': [75000., 0., 0.], 'SubhaloVel': [0., 0., 0.],
        'SubhaloMassType': [0., 0., 0., 0., 1., 0.], 'SubhaloVelDisp': 200.,
        'SubhaloHalfmassRadType': [0., 0., 0., 0., 1000., 0.],
        'SubhaloStarMetallicity': .02}


@pytest.mark.parametrize('source', ['api', 'raw_csv'])
def test_tng_api_target_and_observations_convert_exactly_once(tmp_path, monkeypatch, source):
    loader = _tng_loader(tmp_path)
    group = {'id': 7, 'Group_M_Crit200': 100., 'GroupNsubs': 1}
    monkeypatch.setattr(loader, '_api_get', lambda endpoint, params=None:
        {'results': [group]} if endpoint.endswith('/groups/') else dict(group))
    monkeypatch.setattr(loader, '_fetch_group_subhalos', lambda group_id: [_api_subhalo()])
    if source == 'api':
        raw = loader._fetch_from_api()
    else:
        path = tmp_path / 'central.csv'
        pd.DataFrame({'group_id': [7], 'halo_mass_log': [12.]}).to_csv(path, index=False)
        raw = loader._load_from_raw_csv(path)
    halos = loader._group_into_halos(loader._parse_all_subhalos(raw), raw)
    assert len(halos) == 1
    assert halos[0].halo_mass == pytest.approx(1.4762326542663125e12)
    assert halos[0].log_halo_mass == pytest.approx(12.169154810341062)
    subhalo = halos[0].subhalos[0]
    assert subhalo.position[0] == pytest.approx(110.71744906997343)
    assert subhalo.half_mass_radius == pytest.approx(1.4762326542663125)


@pytest.mark.parametrize('source', ['api', 'raw_csv'])
def test_tng_api_rejects_unsupported_epoch_before_any_request(tmp_path, monkeypatch, source):
    loader = _tng_loader(tmp_path, snapshot=50)
    def unexpected_request(*args, **kwargs):
        pytest.fail('unsupported API epoch must be rejected before a request')
    monkeypatch.setattr(loader, '_api_get', unexpected_request)
    with pytest.raises(ValueError, match='snapshot 99.*manifest'):
        if source == 'api':
            loader._fetch_from_api()
        else:
            path = tmp_path / 'central.csv'
            pd.DataFrame({'group_id': [7]}).to_csv(path, index=False)
            loader._load_from_raw_csv(path)


def test_tng_invalid_row_fails_closed_instead_of_shifting_group_membership(tmp_path):
    loader = _tng_loader(tmp_path)
    frame = _legacy_rows()
    second = frame.copy()
    second['subhalo_id'] = 2
    second['group_id'] = 8
    frame['stellar_mass'] = 'malformed'
    raw = (pd.concat([frame, second], ignore_index=True), {})
    with pytest.raises(ValueError, match='row.*0'):
        loader._parse_all_subhalos(raw)


def _legacy_rows():
    return pd.DataFrame({'subhalo_id': [1], 'group_id': [7], 'stellar_mass': [5.],
        'velocity_dispersion': [200.], 'half_mass_radius': [1500.], 'metallicity': [.02],
        'pos_x': [75000.], 'pos_y': [0.], 'pos_z': [0.], 'halo_mass': [1e12]})


def test_clustered_legacy_schema_converts_low_log_masses_without_magnitude_guessing(tmp_path):
    path = tmp_path / 'legacy.csv'
    _legacy_rows().to_csv(path, index=False)
    loader = _tng_loader(tmp_path)
    raw = loader._load_from_clustered_csv(path)
    subhalo = loader._parse_all_subhalos(raw)[0]
    assert subhalo.stellar_mass == 1e5
    assert subhalo.position[0] == pytest.approx(110.71744906997343)
    assert subhalo.half_mass_radius == pytest.approx(2.2143489813994686)


def test_saved_normalized_clustered_csv_roundtrip_does_not_reconvert(tmp_path):
    loader = _tng_loader(tmp_path)
    df = loader._subhalos_to_dataframe([{**_api_subhalo(), 'group_id': 7, 'halo_mass': 1e12}])
    path = tmp_path / 'normalized.csv'
    loader._save_clustered_csv(df, path)
    raw = loader._load_from_clustered_csv(path)
    subhalo = loader._parse_all_subhalos(raw)[0]
    assert subhalo.position[0] == pytest.approx(110.71744906997343)
    assert subhalo.half_mass_radius == pytest.approx(1.4762326542663125)
    assert subhalo.stellar_mass == pytest.approx(1.4762326542663126e10)


def test_explicit_clustered_units_support_normalized_lengths_with_log_stellar_mass(tmp_path):
    path = tmp_path / 'explicit.csv'
    df = _legacy_rows()
    df['pos_x'] = 110.
    df['half_mass_radius'] = 2.
    df.to_csv(path, index=False)
    loader = _tng_loader(tmp_path, clustered_units={'position': 'Mpc', 'radius': 'Mpc',
        'stellar_mass': 'log10_Msun', 'halo_mass': 'M_sun'})
    subhalo = loader._parse_all_subhalos(loader._load_from_clustered_csv(path))[0]
    assert subhalo.stellar_mass == 1e5
    assert subhalo.position[0] == 110.
    assert subhalo.half_mass_radius == 2.


def test_strict_periodic_manifest_is_rejected_before_catalog_access(tmp_path):
    catalog = _catalog(tmp_path / 'not_opened.h5')
    catalog['conventions'].update(periodic_box=True, box_size={'value': 75000., 'unit': 'ckpc_per_h'})
    with pytest.raises(ValueError, match='periodic.*minimum.image'):
        load_astronomy_manifest(_write_manifest(tmp_path, [catalog]))

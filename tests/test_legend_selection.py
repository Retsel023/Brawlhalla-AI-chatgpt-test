"""Legend requests require explicit capabilities and acknowledgments over sockets."""
import unittest

from session_bridge import SessionBridgeClient
from session_profiles import make_profile
from state_bridge import BridgeError
from team_control_bridge import TeamControlClient
from test_session_bridge import SessionProviderFixture
from test_team_control_bridge import TeamControlProviderFixture


class LegendSelectionTests(unittest.TestCase):
    def provider(self, *, teams=False, capability=True, wrong_at=None):
        profile = make_profile('bot_training', match_format='2v2' if teams else '1v1')

        def alter(request, response):
            if request['command'] == 'hello':
                if capability:
                    response['capabilities'].append('legend_selection')
            else:
                response['legend'] = 'Orion' if request['command'] == wrong_at else 'Hattori'
            return response

        provider_type = TeamControlProviderFixture if teams else SessionProviderFixture
        provider = provider_type(profile, alter=alter)
        self.addCleanup(provider.close)
        client_type = TeamControlClient if teams else SessionBridgeClient
        client = client_type(port=provider.port, expected_build='11.00.36')
        self.addCleanup(client.close)
        client.connect()
        return profile, provider, client

    def test_unsupported_legend_selection_sends_no_setup_or_actions(self):
        _, provider, client = self.provider(capability=False)
        with self.assertRaisesRegex(BridgeError, 'legend_selection'):
            client.request_legend('Hattori')
        self.assertTrue(provider.released.wait(1))
        self.assertEqual([request['command'] for request in provider.requests], ['hello', 'release'])

    def test_acknowledged_selection_is_sent_with_match_setup_for_both_protocols(self):
        for teams in (False, True):
            with self.subTest(teams=teams):
                profile, provider, client = self.provider(teams=teams)
                client.request_legend('hattori')
                client.configure(profile)
                client.next_episode()
                for command in ('configure', 'next_episode'):
                    request = next(item for item in provider.requests if item['command'] == command)
                    self.assertEqual(request['legend'], 'Hattori')
                with self.assertRaisesRegex(BridgeError, 'before session configuration'):
                    client.request_legend('Orion')

    def test_bad_configure_or_game_state_acknowledgment_closes_provider(self):
        for teams in (False, True):
            for wrong_at in ('configure', 'next_episode', 'step'):
                with self.subTest(teams=teams, wrong_at=wrong_at):
                    profile, provider, client = self.provider(teams=teams, wrong_at=wrong_at)
                    client.request_legend('Hattori')
                    with self.assertRaisesRegex(BridgeError, 'legend'):
                        client.configure(profile)
                        client.next_episode()
                        client.step(('light',), 0.1)
                    self.assertTrue(provider.released.wait(1))


if __name__ == '__main__':
    unittest.main()

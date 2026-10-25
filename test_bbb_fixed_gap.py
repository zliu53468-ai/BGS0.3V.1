"""Fixed display threshold, state isolation and zero impact on wager decisions."""
import ast,copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import bbb_final57_runtime as runtime
ROOT=Path(__file__).resolve().parent
class FixedGapTests(unittest.TestCase):
    def test_exact_threshold_both_directions(self):
        for b,expected in [(.51,'B'),(.5099,'Skip'),(.49,'P'),(.4901,'Skip'),(.5,'Skip')]:
            with self.subTest(b=b):
                result=runtime._run({'operation':'display_direction','forecast':{'available':True,'directional':{'B':b,'P':1-b},'direction':'B' if b>=.5 else 'P'}})
                self.assertEqual(result['value'],expected);self.assertEqual(result['gap'],.02)
        self.assertIsNone(runtime._run({'operation':'display_direction','forecast':{'available':False}})['value'])
    def test_profile_is_fixed_and_ui_has_no_selector(self):
        profile=json.loads((ROOT/'cardcount_profile.json').read_text(encoding='utf-8'))
        self.assertEqual(profile['display_gap'],.02)
        self.assertNotIn('forecastGap',(ROOT/'static/liff.html').read_text(encoding='utf-8'))
    def test_local_learning_is_causal_scoped_and_rewindable(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'BGS_BBB_STATE_FILE':str(Path(tmp)/'state.db')}):
            scope={'scope_key':'fixed-gap/user-A/table-1/shoe-1'}
            runtime.apply_final_prediction('B',scope)
            value=runtime.apply_final_prediction('BBPT',scope)
            self.assertEqual(value['browser_calibration']['seen'],3)
            self.assertEqual(value['browser_calibration']['ties'],1)
            again=runtime.apply_final_prediction('BBPT',scope)
            self.assertEqual(again['browser_calibration']['seen'],3)
            other=runtime.apply_final_prediction('BBPT',{'scope_key':'fixed-gap/user-B/table-1/shoe-1'})
            self.assertEqual(other['browser_calibration']['seen'],0)
            rewind=runtime.apply_final_prediction('B',scope)
            self.assertEqual(rewind['browser_calibration']['seen'],0)
    def test_panel_only_changes_local_display_copy(self):
        tree=ast.parse((ROOT/'runtime_app.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fixed_forecast_panel')
        captured={}
        def original(user,session):captured.update(session);return session
        ns={'_ORIGINAL_SCREEN_RESULT_PANEL':original}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'display_boundary','exec'),ns)
        session={'access_until':1234,'permanent_access':False,'screen_last_prediction':{
            'action':'Skip','skip':True,'bet_allowed':False,'bet_amount':0,
            'bbb_final57':{'active':True,'forecast_direction':'B','next_hand_forecast':{'available':True,'directional':{'B':.51,'P':.49},'probabilities':{'T':.1}}}}}
        old=copy.deepcopy(session);ns['_fixed_forecast_panel']('user',session)
        self.assertEqual(session,old)
        shown=captured['screen_last_prediction']
        self.assertEqual(shown['internal_action'],'B')
        self.assertEqual(shown['banker_rate'],51.)
        self.assertFalse(shown['bet_allowed']);self.assertEqual(shown['bet_amount'],0)
        self.assertEqual(captured['access_until'],1234)
        self.assertFalse(captured['permanent_access'])
if __name__=='__main__':unittest.main()

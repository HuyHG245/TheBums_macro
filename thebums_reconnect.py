"""Image-driven reconnect state machine; *only* the macro owner supplies UI input.

This module deliberately does not import tkinter or Roblox-specific APIs. Matching
is deterministic and testable with previously captured screenshots. Every action
has an independently observed gate before the next action is permitted.
"""
from __future__ import annotations

import dataclasses
import os
import time
from typing import Callable, Optional

import cv2
import numpy as np


class ReconnectError(RuntimeError):
    pass


class NewDisconnect(ReconnectError):
    pass


@dataclasses.dataclass(frozen=True)
class Match:
    name: str
    x: int
    y: int
    confidence: float
    width: int
    height: int
    source: str = "opencv"


# ROIs are fractions of the frame dimensions, not screen pixels. They are
# restricted by the full screenshots the user provided, to reject duplicate
# labels on different panels (e.g. Redeem vs VIP Servers or two START buttons).
REGIONS = {
    'slot_a': (.20,.20,.43,.45),
    'select': (.18,.52,.45,.72),
    'play_small': (.34,.46,.65,.65),
    'codes': (.32,.66,.67,.86),
    'lobby': (.30,.35,.70,.78),
    'ps_input': (.01,.76,.17,.98),
    'generate': (.01,.83,.17,.99),
    'play_large': (.72,.62,.99,.94),
    'missions': (.10,.24,.40,.86),
    'outskirts': (.58,.47,.78,.74),
    'outskirts_selected': (.28,.25,.60,.45),
    'create': (.40,.71,.66,.83),
    'start': (.46,.70,.69,.83),
}
# New Raid markers are deliberately separate from the original Mission assets.
RAID_REGIONS = {
    'raids': (.32,.27,.50,.44),
    'attack_selected': (.29,.26,.56,.37),
    'female_raid': (.60,.54,.76,.63),
}
RAID_ASSETS = tuple(RAID_REGIONS)
THRESHOLDS = {name:.82 for name in REGIONS}

THRESHOLDS.update({'reconnect':.90, 'slot_a':.86, 'select':.85,
                   'start':.88, 'generate':.85, 'ps_input':.87})


# The English Roblox disconnect/Home UI anchors below were extracted from the
# user-supplied recording (22 Nov 2026). They are compact binary glyph masks
# held inline so that all original recognition images stay byte-for-byte intact.
# OpenCV alone handles this navigation; there is no OCR, API or remote request.
# If Roblox changes these labels the route MUST fail closed (no blind click).
_CLIENT_UI_B64 = {
    'disconnect_title': 'iVBORw0KGgoAAAANSUhEUgAAAI8AAAAhCAAAAADFHqfRAAAA80lEQVRYw+1V7Q7DIAjsNb7/K9/SxSIofixZarJxP5ozICBfPY5A4B/AJ3w0TlIbAjInvvm4RWNnL2jsqUqyR5TM7AnIyQ92dm3yy4275FQNlVkhMPw6QBqxsOOtxjv5rU43GRTXkE9W4q1OXdghV7fYXGanGGl5LKCIDJ+yXHmhvgUxBCfcSf94W4jeSrCvg33ElQy6zysSrPaP+KCkCB+0O7uzwdnUnOPNDHhL9PH5gqaUkbNkkFYzqGuSwX52aFUtjqPxhoeOhNP81H1WxwZNJg2EmqGVLO1nWFo+lyEhgwSJSnVLS7b/CQKBQCAQ+F28ABtyVUI/bHCmAAAAAElFTkSuQmCC',
    'leave_button': 'iVBORw0KGgoAAAANSUhEUgAAAFMAAAAkCAAAAADQ/lmlAAAAfUlEQVRIx+2U3Q6AIAhG/Vrv/8qnmWKu1irsyuEFIIMz+Zki/X6WFMxgBjOY0uy1ry9ichMwpWwWWf0OpqVnLUFpMwfWUzuVTDMLyi6e2tUb4up3MGkIdmFtZGDuOsFtczWyS2r5uvGf4nl4Hv0uUYdNv2MfmPHXBTOYszE3syQgV0TkLe8AAAAASUVORK5CYII=',
    'home_game_title': 'iVBORw0KGgoAAAANSUhEUgAAAJoAAAAqCAAAAABuXh2+AAABbElEQVRYw+1X0ZKDIAxknf7/L+9ZQ4EEFLRwN3OTfUBLQ1hCAmsIDodjIcivhr+0sxBwPIu2mjCgwyg+xYyYQ20K8KGPMC9qu0fKsuWNMk0KQJotRy72iT2q+FonTbsTbPWC44qRfheN3iyZl8H+qxxqJyd2vailETgSJLbZHyrDdx/IkoDlpp2c2g3lWpW3NlIlSVnCSLVOqNBWQfLrY+SJk02N39FKF5j9Lnd0LCB4UK9b7YQqZfdGjk6kt8zt6OtMqpw8pYaSH2O7NzGUaLBAl5lx4nA4HP9cyQ7eobQHb0fgcvEh+qrl2l1J+yvUDoXDpCeJkEVqKR4L/cooXm9r2AfXe6lheKEvoLTufQ17+9sg3dhRu+YbnDa1QB3IK607J2qilZm/M0yEzhIOq8vAVCn/UslsLSFpj4teBs3MsOsySHoWTS3LDw2Gca07gRrkIKBKvVLLIm0z0lWAFcwcDodjgXBdKGEH8AOBXKgkUyC3lAAAAABJRU5ErkJggg==',
    'game_detail_title': 'iVBORw0KGgoAAAANSUhEUgAAAawAAAAvCAAAAACa8KGyAAACRElEQVR42u2Z63bjIAyEPT55/1eenr2cBoME4ua4u/P9aFOXKCOBJCDHIYQQQgghhBBCCCGEEOLRIH9A62EnnDHBQuCEOfqOrnSG8zGL8LK8IywVNynatwbwIDlDnP9RFeFPd+Blu/XBFAKbpfr5zeSWyeJTPOe6EODbNeJfyyy8Z4zfP3B5/Z5W5NOM6maFefhwsVltDswscsRU8c7MIDxXkg5lyKGtqk9Wb89iNMPI6++DzhutuboM4lhGFwLipox3Fkk9WmUs21xWs8Y2GPQf0B9lDOJgZOhPX8sUncRnzZVuWdMehsogkhaPJL/T179GIO0DSQWBWwSTmpF0D0T9qApom0o/HVlgkRkoVTbkrPGwL7MY3mkAeZFD+tzbIqCojBis56UA1xR/Y5VmVnYkgz1mnYe7zll0HqF5mbBwe4bQiEIFsm4zIW23h2UZ5PD8dA0g1i+Q6GQyPN0zKjecE16e/9XP+vhxjH2zxNvjeuMNRtfZ9eOH59ityLXj5zcm3CNvLefI4g1cL6CxB8J0YuGG6D4s484sAn9ZsIjy2UJkGezMQvoz8t4s1lS2TrfbPTxDq9gRYVwfsH3vwMmUcgT0GiHdUNdUsl59uHM1vuzSkhZ5Zhtc/Plv0RPyZo3s6hRJDGa+CrQExLuWKfVoq7Q/8lqP0zG4P7NQLXYwejYihRDDtcwTMFMI4fyFymm81eO2dLvzGJktFMdyIBA9dIfYjcXEvQBMGWipRGD1TnsohBBCCCGEEEIIIYQQQoh7+ALC6uwzbzcgDwAAAABJRU5ErkJggg==',
}
_CLIENT_UI_B64['leave_256'] = 'iVBORw0KGgoAAAANSUhEUgAAAEAAAAATCAAAAAAFGFRbAAABhUlEQVQ4EaXBP2gTYRjA4d+bDi2CyiE2WgK6CCaglkpBePk+aHuCglAdO3cSg3+GkqKuB4KISzo4tOLcHHapWIkgElxUvMMhQ5CUUKGCrYPWQbF653If3cI9jyj5iJKPKPmI4rAmoE+iOKwJ6JMoDmsC+iSKw5qA/ypT3nYr4si5E/u+PuteLtWBa5+eU5nytlsRDlEc1gSkRv1wY2Qm7Izvf/fLP/3g6Gz9C4evL2yO+uHGyEzYISOKw5qARKG20gbfWyZxYO5x9+bHJpOVeqG20gbfWyYjisOagESxSmp9kWOnSoeGGvHkmYfceN8qVkmtL5IRxWFNQKJYXeqSujj+pvNtrhF7tx7tXr3/vVhd6rKHKA5rAhID89EqiaHbTz8weLcRM9vbLT1hYD5aZQ9RHNYEpMam16JCuf3zztuX3qXjYcxZ8+d1DGPTa1Gh3P5BRhSHPU9ic4HyxPBOr7l18sLBzy+uvIoZrP299xsoTwzv9JpbZETJR5R8RMlHlHxEyecfbZR1dcUxfDkAAAAASUVORK5CYII='
_CLIENT_UI_B64["body_256"] = 'iVBORw0KGgoAAAANSUhEUgAAAWEAAAA/CAAAAAAUZyY4AAAPMElEQVR4Ae3BfVCU94HA8e+zazSJJBFd/QGxgxlj66oI61Zj+7jLIGuKsYRftkDO3Glt7q7ejdt0OjdhadPtP89s73qZDq3EnM+dWmORpImQDQoKNag548uVl6UCK2eML2cMEdQGMBefheUeSLg0HTuxvdsu0+HzUVQmxJWiMiGuFJUJcaWoTIgrRWVCXCkq8eaa+eowv5fDqxn8/5G2bcSbFDq3T1EZ43YFuZWZT2kGf5SZT2kGFKY+P8Tv5fBqBrcw8ynN4A8nbduIE/cqhj9472QnUujcPkVljNsV5FZmb9QM/iizN2oGn8Hh1QxuYfZGzeAPJ23biBO3K2hJWvzwq61S6Nw+RWWM2xXkVuZu0Az+KHM3aAafIetrmsEtzN2gGfzhpG0bceJ2BYEnpv6bFDq3T1EZ43YFGWXPtfWfOB7jc3kpA6cbVy63wM6zmB70iP6Og0P2XFv/ieMxZOxC9rTu6l5k7EL2tO7qXliQm3ztaBge9Ij+DutDFth5Vgod7Lm2/hPHY8jYhexp3dW9jMhcmXT5HVUz1g9WQp69DBbkJl87GgZWL7fAzrP2XFv/ieMxTCnL593dW3cOGbuQPa27uhfT+sFKyLOXIWMXsqd1V/cibduwffONozJ2IXtad3Uv2HNt/SeOx1bk/HCIJx74xxiPD78sYxeyp3VX9zJiQW7ytaNh5NDllX26HLq8sk/HnmvrP3E8hhy6vLJPZ5TbFQRWzy+TQocFucnXjoZJWT7v7t66c3wuL2XgdONNFuQmXzsa5rcoKmPcriAjsgpqIqlF7XUWf0tjcmbjkMOrGYzIKKxvvXf+kayCmkhqUXsdMrPlkLJuYBcys+WQsm5gF1meqktpa6vOZBTWt947/4jDqxkghU5WQU0ktai9DpnZckhZN7AL05JH97VPfyRdM9YPVkKevYwsT9WltLVVZwCHVzPIKqiJpBa112Faek+T4Vn845sys+WQsm5gF6b1g5WQZy9DZrYcUtYN7ELatt39zbN7kZkth5R1A7vIKqiJpBa118361rYLllLj5YuKv6FFZrYcUtYN7MKU5am6lLa26oycF2mMRuW8SGM0mlVQE0ktaq9Dzos0RqOMcruCwAZ2SqGT5am6lLa26szSe5oMz+IfR/0tjcmZjUNZnqpLaWurzvAJRWWM2xXEZC1pPQAZReW/eab+OCaHVzMwWUs6agBrSesByCgq75EpW2FFdhCZshVWZAct/lAEPMnVJR01mBxezQApdGtJ6wHIKCrvkSlbYUV2ELCUdOwFh1cz1g9WQp69zOIPRcCT/Arg8GqGtaT1AGQUlffwkXuf/tnbMmUrrMgOYlo/WAl59jJkylZYkR1E2n624WZlDJmyFVZkB60lrQcgo6i85x9aG9PXXLrxesqmZ/tkylZYkR0ELP5QBDzJr8j0zcMg0zcPYy1pPQAZReU9Mn3zMB9zu4LKPUvd2y9KoVv8oQh4kl/BdO/TP/uvZ+qPAxZ/KAKe5Ff4hKIyxu0KYhK+ii5I8oeal65+ty1s4PBqBibhq4wAwlfRBUn+ULMUOjhlACl0cMqA8DHi/D5fZQSTw6sZIIUufBVdkOQPNUuhg1MGgJlPvdQBDq9mrB+shDx7mfAx4vx2wOHVDOGr6IIkf6gZU3rG7Bl37mmTQgenDGBaP1gJefYypNDBKQNI2/WsX7SDFDo4ZUD4KrogyR9qzhfbcie9++WtX3aWI4UOThkAhI8R57dLoQNS6CB8FV2Q5A81S6Ezxr0KBt49cgEpdOFjxPntpGfMnnHnnralq99tCxvCx4jz2/mEojLG7QpiEr6KLphaGmrm7oyHpmy77vBqBibh230aEL6KLphaGmqWQgenDCCFDk4ZEL4d5zAJ3+7TmBxezQApdOGr6IKppaFmKXRwygAw/Tu7T4PDqxnrByshz14mfDvO8TGHVzOEr6ILppaGmoHVS4+duf70njYpdHDKAKb1g5WQZy9DCh2cMoB0XG9buvlDpNDBKQPCV9EFU0tDzfP/4ocbXu8u+ZH36n6k0MEpA4Dw7TjHCCl0QAodhK+iC6aWhpql0BnjdgUZJYUufDvOMWL10mNnrj+9p427Mx6asm2yb8c5foeiMsbtCmKy+psaYFFxeQ8w6dutjYuLgh9ispa27Aes/qYGWFRc3iOFDk4ZQAodnDJgLQ3XYrKWtuzHtLgo+CFIoVv9TQ2wqLi8RwodnDIAWAJvHgSHVzOKk3bAY3PKrKXhWj62uCj4odXf1ACList74M7vvdrKlO/vaZNCB6cMYCpO2gGPzSlDCh2cMoBcsPX6xvdeRQodnDJg9Tc1wKLi8p7J33258J+if3/00V+8hRQ6OGUAsJaGaxkhhQ5IoYPV39QAi4rLe6TQGeN2BRklhW4tDddiuvN7r7Yy5ft72oBJ3249Uhqu5XcoKmPcriAjHPk1kbTCztr7H27oTvnG3raUTdUdyk1g2SN1p5KWNyzIr4mkFXbWIoUOThlACh2cMsCSgvqwxR4ZWPZI3amk5Q3Jm6o7lJtS6DjyayJphZ21SKGDUwYwPbxsz9n75QzNUD3bryz/4nAZSwrqwxZ7ZABI2VTdoSzIr4mkFXbWApZnfvV68lfnVLVJoYNTBjCpnu1Xln9xuAwpdHDKANK2jbSNP39LCh2cMoAjvyaSVthZCxvuiO5k1Zy0H0aRQgenDGBaUlAfttgjA1LogBQ64MiviaQVdtYihQ7FaVuigNsVZNSaueXDSwrqwxZ75INnfvV68lfnVPU+3NCd8o29bUsK6sMWe2SATygqY9yrMHVvYWGOrf/ksdgUNfO+vqY3IHeZ9bVTmBZnz3i/8/DNhTm2/pPHYkihg1MGkEIHpwyAPWfWjYsHr7I4e8b7nYdv5i6zvnZKCh0W5tj6Tx6LIYUOThnANGlV5uS3mtZpxmT5eePNPk8Z2HNm3bh48Cqm3GXW104tzLH1nzwWwzQ/7753Gh473CaFDk4ZwDRZft54s89ThhQ6OGUAadsGa+Y/t1ro4JQBWJhj6z95LAZfXt3w78z567M7QQodnDLACHvOrBsXD16VQgek0DEtzLH1nzwWQwodHk/dEgXcriCj0p8I78eeM+vGxYNX5+fd907DY4dPq5n39TW9AfacWTcuHrzKJxSVCXGlqEyIK0VlQlwpKhPiSlGZEFeKyoS4UlQmxJWiMiGuFJUJcaWo/NmY+ZRmMO4oKn82Zm/UDMYdReXPxtwNmsG4o6iMH3Lo8so+nQW5ydeOhuFBj+jvODhkz7X1nzgeQ8YuZE/rru5Fxi5kT+uu7oUFucnXjobhQY/o77A+ZIGdZxlnFJXxQ86LNEajWZ6qS2lrq85kFNa33jv/SFZBTSS1qL0OmdlySFk3sAuZ2XJIWTewiyxP1aW0tVVnMgrrW++df8Th1QzGHUVl/JDpm4ex+EMR8CRXl3TUANaS1gOQUVTeI1O2worsIDJlK6zIDlr8oQh4kqtLOmowObyawbijqIwfUuggfIw4v89XGQGEr6ILkvyhZil0cMoAUujglAHhY8T5fb7KCCaHVzMYdxSV8UMKHYRvxzlMwrf7NCB8FV0wtTTULIUOThlACh2cMiB8O85hEr7dpzE5vJrBuKOojB9S6GAtDddispa27Aes/qYGWFRc3iOFDk4ZQAodnDJgLQ3XYrKWtuzHtLgo+CHjjqIyfkihA0sK6sMWe2Rg2SN1p5KWNyzIr4mkFXbWIoUOThlACh2cMsCSgvqwxR4ZWPZI3amk5Q3Jm6o7lJvFaVuijCOKyvghhY7JnjPrxsWDV1mcPeP9zsM3F+bY+k8eiyGFDk4ZQAodnDIA9pxZNy4evMri7Bnvdx6+mbvM+tqpx1O3RBlHFJUJcaWoTIgrRWVCXCkqE+JKUZkQV4rKhLhSVCbElaIyIa4UlcQoOPU2/zcOr2ZwW2xrXiBhFJWESP3Lnwy6V2EyND6DstQ5c+js4W4+zeHVDD4tPXf2h6frb5L1NUw/GmB63tzh/zzQt/FoB4miqCTEY9eO4HYFuQ3K4/fve/vO7KH9fJrDqxl8yvRN9eFpxe+9wpeW/ZQRU33NJ62uk1cylm0nURSVRLjju+XXcbuC3Ial+c9d4RayvqYZfFrSACws1gZXPrCdEWum/5wRd5RuuUaCKCqJ8EDhs+B2BRklhy6v7NPl0OWVfTr2XFv/ieMx5NDllX06sOnqS3zMnmvrP3E8Bpkrky6/o2oGC3KTrx0NU5y2JcpHHvz6P/evueclRvj3/5pRT7Y1kyCKSiKsmFMBbleQUXJepDEalfMijdFoVkFNJLWovQ45L9IYjcKkHxx8g49kFdREUova61jy6L726Y+ka0aWp+pS2tqqM4+nbonykYIvPDtcONsy9d1fXkjyVy6Z23fsP2D1pL0kiKKSCF+5KwTuVZieew+ZvnkYZPrmYawlrQcgo6i8R6ZvHsaU5K9uZZS1pPUAZBSVXy3p2AsOrzboD0XAk/wK/2v23/7yKA9Ob5/08ILnLd/qrbvwhaLdXbjvf5EEUVQSIX+oDtyuIKOk0AEpdBC+ii5I8oeapdAZMekHrx9hlPBVdEGSP3TxqZc6wOHVkn2MOL+dMVP/7oN/HWLEJP+vWr5d0QXrPqjiIftOEkRRSYSv3BUCtyvIKCl0QAodhK+iC6aWhpql0Bn1rd4XGSV8FV0wtTR07ju7T4PDqyX7dpzjt0158r5/eZ+P+C7Vff/ldng0+QXc979IgigqibBiTgW4XUFGSaEDUuhg9Tc1wKLi8h4pdEZ9afVzV4C7/tvqb2qARcXlVwNvHgSHVxsqDdfyWyZ/PXXnRWDKTbizpPHo31yrhiev7GP1HTUkiKKSCA8UPgtuV5BRUuiAFDrgyK+JpBV21iKFzijLX6W9evYul/0nQ478mkhaYWctDy/bc/Z+OUMzlhTUhy32yEBx2pYoTF7/ucouIGPV3vNJ+dOfNx5Yv7djYf7zPTzZ1kyCKCqJcMd3y6/jXsWIH/9GCh2QQse0MMfWf/JYDCl0PmL9kmOGcf7Qu7Awx9Z/8liMSasyJ7/VtE4zsOfMunHx4NXHU7dEwf7E0OBkhchLD2XZom819MF8z4wrB85xR+mWaySIopIQ3qtH+NNZtHwbiaKoJETqEz8d5E/mm292kCiKSmLIX7/Nn4ptzQskjKIyIa4UlQlxpahMiKv/AZXa/BBEk9EOAAAAAElFTkSuQmCC'
_CLIENT_UI_B64["title_256"] = 'iVBORw0KGgoAAAANSUhEUgAAAI8AAAAiCAAAAABDitV/AAAEvUlEQVRYCc3BbUxVZQDA8f9z5RKldhL0+k4fCrSc0/vBtfmUm+0kimtpKjo5XWcOPGSTjMydtqOmE1e+3MnVdGOW00yMPHChm/hSJ70uY86UofmSwBSV1QdnmNwDCnbRD/durm+2nd9PSFxFSFxFSFxFSFxFSFxFSFxFSFxFSFxFSP5D9rtN5fzPfBZGlGRCkuAPQc+dpmPfdwLrX2NBE09KcIJp8xifhRElmZAk+EM8dH3FVcgpaVrm8KR8mW3aPMZnYURJJiQJ/hCBa0NztJQbCzt4sr4bYto8xmdhREkmJAn+EIFmmLyW7Xt5so48bdo8Jr0GI0oyIUnwhwg0A+UvXShA0x0Vz6w3h3XWl7eBJ3d6Fo3bL5Mye1pmx297LoOmV+8rnJB2bkMraHr1vsIJaec2tALevGnD24+Xt4Mnd3oWjdsn5z0FOCrevGnD24+Xt8MzC9Xnruz6DCNKMiFJ8IcINANF+fcnP9B0R6UwQNztObGU0onENRalbh5P3L1Vx9H0W/1SgQsFoOm3+qUCFwogNTiOuKaCrpTSicQ1/vE2cY6aGhxHXFNBV9oX2TxkREkmJAn+EIFmYE4xU//RdEfF8u0vH/jWlUMsKKCyonNmXdvid3rK6gYZL9/N+1vTubj+5uLZ5F9F07m4/ubi2eRfpTBwe92ZwStHbalcUEBlRefMujafhWlDYeD2ujODV47aUlkYoLwqXX8VI0oyIUnwhwg0A3lLmdKh6Y7KV1m/bLkBiOqM+hLiPLXK0dUw4htP2bea3jU9xsh9lNSj6V3TY4zcR0m9qBmw9hBMKj31YXVGfQm9fBamjagZsPYQTCo9taxq0K8fgc/CiJJMSBL8IQLNwPvzul5H0x2VMZ8rXAnXdKfXEDxAXEaYjdWA5fuhVNMdFZQIpo2mOyooEUw7I8wjrUtqCB6gl8/CtMkI80jrosPs+Bp8FkaUZEKS4A8RaAZ2vdiwBE13VOj/Zs4LnF32bJjNFnEZYTZWAwcGH1yn6Y4KSgTTRtMdFZQIpp0R5pGWD8JstujlszBtMsI80lJUx/a94LMwoiQTkgR/iEAzTFnJpio03VHx3gN/MMU8Ful/cjlxnlrlyKcwrMKztULTHRWUCKaNpjsqKBFM21OrbKqilyfS/+RyeqXXsPoonlplUxW9xMF+tgkjKjCiJBOSBH+IwLWhU7U+LYu60HRH9e64srcte6t3zeHiOVTsj+Xet96b3xM8MtAYE5t7S9MdFZQIpo2mOyooEUybovxY2YnYaLmN4jlU7I/l3rc8dp/o2rRbRfmxshOx0XIbq97o2fDTkE9GYURJJiQJ/hAPXV1+EzTdUccHvcS1LurouzWLuLszHgTHEte95kc03VFBiWDaaLqjghLBtEkLjqVX8em+W7OIuzsjtlqFezme4Fh6FZ8etrM/0PkgzYiSTEgS/CHoudPyc00XoOmOyvPzX1H+bNj5F6TNU4d3Nuw+j3duzohYw57fQdMdFZQIpo2mOyooEUwbvLOmZnZfqjwGafPU4Z0Nu8/Tb+mk1Osr2ryzpmZ2X6o8BplF/pSzZR+PN6IkExJXERJXERJXERJXERJXERJXERJXERJXERJXERJXERJX+ReCW80IPIqoxQAAAABJRU5ErkJggg=='
_CLIENT_UI_CACHE = {}


def _ui_template(name):
    if name not in _CLIENT_UI_CACHE:
        from base64 import b64decode
        encoded = np.frombuffer(b64decode(_CLIENT_UI_B64[name]), dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if decoded is None or decoded.std() < 5:
            raise ReconnectError('Embedded Roblox UI anchor could not be decoded')
        _CLIENT_UI_CACHE[name] = decoded
    return _CLIENT_UI_CACHE[name]


def _find_ui_text(frame, name, roi, threshold=.82, scales=(1.0, .9, 1.1, .8, 1.2)):
    """Match an anchored, locally embedded UI label with bounded scaling."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return None
    h, w = frame.shape[:2]
    if h < 480 or w < 800:
        return None
    x0,y0,x1,y1 = roi
    xa,ya,xb,yb = max(0,int(w*x0)),max(0,int(h*y0)),min(w,int(w*x1)),min(h,int(h*y1))
    if xb<=xa or yb<=ya:
        return None
    image = cv2.cvtColor(frame[ya:yb,xa:xb,:3], cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 185, 255, cv2.THRESH_BINARY)[1]
    original = _ui_template(name)
    best=None
    for scale in scales:
        tw=max(8,round(original.shape[1]*scale)); th=max(8,round(original.shape[0]*scale))
        if image.shape[0]<th or image.shape[1]<tw:
            continue
        tpl = original if scale==1.0 else cv2.resize(
            original,(tw,th),interpolation=cv2.INTER_NEAREST)
        # A nearly uniform patch must never be taken as a positive match.
        if tpl.std()<5:
            continue
        _,score,_,at = cv2.minMaxLoc(cv2.matchTemplate(
            image,tpl,cv2.TM_CCOEFF_NORMED))
        if best is None or score>best.confidence:
            best=Match(name,xa+at[0]+tw//2,ya+at[1]+th//2,
                       float(score),tw,th)
    return best if best is not None and best.confidence>=threshold else None


def find_scaled_reconnect(frame, template):
    """Conservative fallback for resized Reconnect dialogs, irrespective of body text.

    Requires the Disconnected title and the white Reconnect label at the
    expected offset. Does not change the original exact-template path or
    use the Leave-only route. No click is performed here.
    """
    if template is None or frame is None or frame.ndim != 3:
        return None
    scales = (1.0, .975, 1.025, .95, 1.05, .925, 1.075, .90, 1.10,
              .875, 1.125, .85, 1.15, .80, 1.20)
    title = _find_ui_text(frame, 'disconnect_title', (.35,.30,.65,.53), .78, scales)
    if title is None:
        return None
    h,w = frame.shape[:2]
    tw = title.width
    xa = max(0, int(title.x + .18 * tw))
    xb = min(w, int(title.x + 1.08 * tw))
    ya = max(0, int(title.y + .95 * tw))
    yb = min(h, int(title.y + 1.65 * tw))
    crop = frame[ya:yb,xa:xb,:3]
    if not crop.size:
        return None
    # Include the expected scale inferred from the title and nearby scales.
    base = tw / _ui_template('disconnect_title').shape[1]
    best = None
    for scale in (base, base-.025, base+.025, base-.05, base+.05):
        t = cv2.resize(template, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC if scale>1 else cv2.INTER_AREA)
        bh,bw = t.shape[:2]
        if bh > crop.shape[0] or bw > crop.shape[1]:
            continue
        _,score,_,(x,y) = cv2.minMaxLoc(cv2.matchTemplate(crop,t,cv2.TM_CCOEFF_NORMED))
        cx,cy=xa+x+bw//2,ya+y+bh//2
        if not (.42*tw <= cx-title.x <= .85*tw and
                1.05*tw <= cy-title.y <= 1.55*tw):
            continue
        patch = frame[cy-bh//2:cy+bh//2,cx-bw//2:cx+bw//2,:3]
        if not patch.size or np.mean(np.all(patch>185,axis=2)) < .60:
            continue
        if score >= .85 and (best is None or score>best.confidence):
            best = Match('reconnect',cx,cy,float(score),bw,bh,'opencv_scaled_dialog')
    return best


def find_error_256_leave(frame):
    """Recognize the specific two-button Error 256 modal, not generic errors."""
    scales = tuple(x / 100 for x in range(65, 151, 2))
    if frame is None or frame.ndim != 3:
        return None
    h,w=frame.shape[:2]
    gray=cv2.cvtColor(frame[:,:,:3],cv2.COLOR_BGR2GRAY)
    xa,ya,xb,yb=int(w*.40),int(h*.32),int(w*.60),int(h*.48)
    crop=gray[ya:yb,xa:xb]
    title=None
    for scale in scales:
        t=cv2.resize(_ui_template('title_256'),None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
        th,tw=t.shape
        if crop.shape[0]<th or crop.shape[1]<tw: continue
        _,score,_,(x,y)=cv2.minMaxLoc(cv2.matchTemplate(crop,t,cv2.TM_CCOEFF_NORMED))
        if title is None or score>title.confidence:
            title=Match('title',xa+x+tw//2,ya+y+th//2,float(score),tw,th)
    if title is None or title.confidence<.78:
        return None
    base=title.width/143
    def local(name, dx, dy):
        tpl=_ui_template(name)
        best=None
        for scale in (base,base*.95,base*1.05,base*.90,base*1.10):
            t=cv2.resize(tpl,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
            th,tw=t.shape
            cx=title.x+dx*scale; cy=title.y+dy*scale
            xa=max(0,int(cx-tw/2-12*scale)); xb=min(w,int(cx+tw/2+12*scale))
            ya=max(0,int(cy-th/2-12*scale)); yb=min(h,int(cy+th/2+12*scale))
            crop=gray[ya:yb,xa:xb]
            if crop.shape[0]<th or crop.shape[1]<tw: continue
            _,score,_,(x,y)=cv2.minMaxLoc(cv2.matchTemplate(crop,t,cv2.TM_CCOEFF_NORMED))
            if best is None or score>best.confidence:
                best=Match(name,xa+x+tw//2,ya+y+th//2,float(score),tw,th)
        return best
    code=local('body_256',0,94)
    leave=local('leave_256',-94,185)
    if code is None or leave is None or code.confidence<.80 or leave.confidence<.86:
        return None
    return Match('leave',leave.x,leave.y,min(title.confidence,code.confidence,leave.confidence),leave.width,leave.height,'opencv_error_256')


def find_leave_disconnect(frame):
    """Distinct Roblox Leave-only disconnect dialog; require *both* text labels.

    Also recognizes the verified Error 256 two-button dialog.
    Other errors retain the existing Leave-only recognition.
    """
    special=find_error_256_leave(frame)
    if special is not None:
        return special
    title=_find_ui_text(frame,'disconnect_title',(.33,.30,.68,.53),.85)
    if title is None:
        return None
    leave=_find_ui_text(frame,'leave_button',(.32,.49,.69,.73),.86)
    if leave is None:
        return None
    h,w=frame.shape[:2]
    dy=leave.y-title.y
    if not (.12*h < dy < .23*h) or abs(leave.x-title.x)>.055*w:
        return None
    # Confirm the target really lies within a pale Leave-button surface.
    y,x=leave.y,leave.x
    patch=frame[max(0,y-6):y+7,max(0,x-35):x+36,:3]
    if patch.size==0 or np.mean(np.all(patch>185,axis=2))<.60:
        return None
    return Match('leave',leave.x,leave.y,min(title.confidence,leave.confidence),
                 leave.width,leave.height,'opencv')


def find_roblox_home_game(frame):
    """Locate the AOT:R card text within Roblox Home / Continue."""
    label=_find_ui_text(frame,'home_game_title',(.02,.20,.97,.97),.84)
    if not label and isinstance(frame, np.ndarray) and frame.ndim == 3:
        # Adaptive capture rescales thin antialiased text. Preserve the original
        # title, ROI and threshold; use a soft white foreground representation.
        from thebums_extra_failsafe import white_mask
        h, w = frame.shape[:2]
        if h >= 480 and w >= 800 and frame.shape[2] == 3:
            xa, ya, xb, yb = int(.02*w), int(.20*h), int(.97*w), int(.97*h)
            roi = white_mask(frame[ya:yb, xa:xb])
            original = _ui_template('home_game_title')
            best = None
            for scale in (1.05, 1.06, 1.07, 1.08, 1.09, 1.10):
                tpl = cv2.resize(original, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_LINEAR)
                th, tw = tpl.shape
                if th > roi.shape[0] or tw > roi.shape[1]:
                    continue
                _, score, _, (x, y) = cv2.minMaxLoc(
                    cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED))
                if score >= .84 and (best is None or score > best.confidence):
                    best = Match('home_game_title', xa+x+tw//2, ya+y+th//2,
                                 float(score), tw, th, 'opencv_soft_home')
            label = best
    if not label:
        return None
    # Move upwards from the text into the matching game's thumbnail.
    cy=label.y-round(95*label.height/42)
    if cy < 20:
        return None
    return Match('home_game',label.x,cy,label.confidence,label.width,label.height)


def find_roblox_game_play(frame):
    """Find the BLUE Play button with the WHITE right-pointing triangle.

    This is the button on the Attack on Titan Revolution detail page in the
    user's recording, *not* a generic PLAY text/icon elsewhere in Roblox.
    A verified game title, a blue button, AND its white triangle are required.
    """
    title = _find_ui_text(frame, 'game_detail_title', (.12, .07, .90, .70), .82)
    if title is None:
        return None

    h, w = frame.shape[:2]
    expected_x = round(title.x - .21 * title.width)
    expected_y = round(title.y + 2.05 * title.height)
    xa = max(0, expected_x - round(.42 * title.width))
    xb = min(w, expected_x + round(.42 * title.width))
    ya = max(0, expected_y - round(.90 * title.height))
    yb = min(h, expected_y + round(.90 * title.height))
    if xa >= xb or ya >= yb:
        return None

    crop = frame[ya:yb, xa:xb, :3]
    # Roblox's bright blue Play button: HSV blue, not gray loading skeletons.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (100, 110, 115), (130, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE,
                            np.ones((3, 3), dtype=np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(blue, 8)
    options = sorted(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]),
                     reverse=True)
    for i in options:
        x, y, bw, bh, area = map(int, stats[i])
        px, py = xa + x + bw // 2, ya + y + bh // 2
        if not (.35 * title.width <= bw <= .85 * title.width):
            continue
        if not (.65 * title.height <= bh <= 1.65 * title.height):
            continue
        if abs(px - expected_x) > .23 * title.width:
            continue
        if abs(py - expected_y) > .65 * title.height:
            continue
        if area / max(bw * bh, 1) < .65:
            continue

        button = crop[y:y + bh, x:x + bw]
        white = cv2.inRange(button, (210, 210, 210), (255, 255, 255))
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            tx, ty, tw, th = cv2.boundingRect(contour)
            if not (.045 * bw <= tw <= .19 * bw):
                continue
            if not (.25 * bh <= th <= .78 * bh):
                continue
            if abs((tx + tw / 2) - bw / 2) > .16 * bw:
                continue
            if abs((ty + th / 2) - bh / 2) > .20 * bh:
                continue
            perimeter = cv2.arcLength(contour, True)
            points = cv2.approxPolyDP(contour, .04 * perimeter, True).reshape(-1, 2)
            if len(points) != 3:
                continue
            # The apex must point RIGHT, with a near-vertical left edge.
            vertices = sorted(points.tolist(), key=lambda p: p[0])
            left_a, left_b, apex = vertices
            if abs(left_a[0] - left_b[0]) > .35 * tw:
                continue
            if apex[0] - max(left_a[0], left_b[0]) < .50 * tw:
                continue
            if abs(left_a[1] - left_b[1]) < .60 * th:
                continue
            if abs(apex[1] - (left_a[1] + left_b[1]) / 2) > .30 * th:
                continue
            return Match('game_play', px, py, title.confidence, bw, bh,
                         'opencv_blue_play_triangle')
    return None


class RobloxLeaveNavigator:
    """Leave -> Roblox Home game card -> game Play -> Slot A verification.

    Deliberately never guesses a fixed Home card position. Every click needs
    two successive OpenCV detections and a fresh pre-click verification.
    """
    def __init__(self, matcher, capture, click, check, progress):
        self.matcher=matcher
        self.capture=capture
        self.click=click
        self.check=check
        self.progress=progress

    def _find(self, frame, state):
        if state=='leave':return find_leave_disconnect(frame)
        if state=='home_game':return find_roblox_home_game(frame)
        if state=='game_play':return find_roblox_game_play(frame)
        if state=='slot':return self.matcher.slot_a_select(frame)
        raise ValueError('Unknown navigation state')

    def _wait(self, state, timeout, stable=2):
        end=time.monotonic()+timeout
        count=0
        last=None
        while time.monotonic()<end:
            self.check()
            hit=self._find(self.capture(),state)
            if hit and last and abs(hit.x-last.x)<20 and abs(hit.y-last.y)<20:
                count+=1
            else:
                count=1 if hit else 0
            last=hit
            if hit and count>=stable:
                return hit
            time.sleep(.3)
        return None

    def _wait_any(self, states, timeout, stable=2):
        """Observe one screenshot at a time and accept any stable next screen.

        Leave may open Roblox Home OR AOT:R detail directly. Never assume
        which one and never click based on a transitional frame.
        """
        states = tuple(states)
        end = time.monotonic() + timeout
        history = {state: (0, None) for state in states}
        while time.monotonic() < end:
            self.check()
            frame = self.capture()
            for state in states:
                hit = self._find(frame, state)
                count, last = history[state]
                if hit and last and abs(hit.x-last.x) < 20 and abs(hit.y-last.y) < 20:
                    count += 1
                else:
                    count = 1 if hit else 0
                history[state] = (count, hit)
                if hit and count >= stable:
                    return state, hit
            time.sleep(.3)
        return None, None

    def _click_verified(self, state, title, next_state, timeout):
        self.progress(0,10,title,'Detecting',1)
        hit=self._wait(state,timeout)
        if not hit:
            raise ReconnectError(f'{title}: {state} not found; no blind click')
        self.check()
        fresh=self._find(self.capture(),state)
        if not fresh or abs(fresh.x-hit.x)>20 or abs(fresh.y-hit.y)>20:
            raise ReconnectError(f'{title}: target moved before clicking')
        self.progress(0,10,title,'Clicking',1)
        self.click(fresh)
        self.progress(0,10,title,'Verifying',1)
        if isinstance(next_state, str):
            next_states = (next_state,)
        else:
            next_states = tuple(next_state)
        observed, next_hit = self._wait_any(next_states, timeout)
        if not next_hit:
            raise ReconnectError(f'{title}: {" / ".join(next_states)} not confirmed')
        self.progress(0,10,title,'Confirmed',1)
        return observed, next_hit

    def run(self, first_frame=None):
        """Handle both Leave outcomes (Home/detail) plus already-open screens.

        The game detail page is never treated as Roblox Home. Every target is
        confirmed over multiple captures, then rechecked immediately before
        input. Slot A is the hand-off to the existing Image Reconnect steps.
        """
        frame = first_frame if first_frame is not None else self.capture()
        if self.matcher.slot_a_select(frame):
            return True
        if find_leave_disconnect(frame):
            next_state, _ = self._click_verified(
                'leave', 'Leave disconnected game',
                ('slot', 'game_play', 'home_game'), timeout=65)
        elif find_roblox_game_play(frame):
            next_state = 'game_play'
        elif find_roblox_home_game(frame):
            next_state = 'home_game'
        else:
            next_state, _ = self._wait_any(
                ('slot', 'game_play', 'home_game'), timeout=25)
        if next_state == 'slot':
            return True
        if next_state == 'home_game':
            next_state, _ = self._click_verified(
                'home_game', 'Select Attack on Titan Revolution',
                ('game_play', 'slot'), timeout=50)
            if next_state == 'slot':
                return True
        if next_state == 'game_play':
            self._click_verified('game_play', 'Play Attack on Titan Revolution',
                                 'slot', timeout=100)
            return True
        raise ReconnectError('Roblox Home / AOT:R game page / Slot A not verified')



class ImageMatcher:
    def __init__(self, image_dir: str, advisor=None):
        self.advisor = advisor
        self.templates = {}
        for name in (*REGIONS, 'reconnect'):
            path = os.path.join(image_dir, 'rc_'+name+'.png')
            tpl = cv2.imread(path, cv2.IMREAD_COLOR)
            if tpl is None:
                raise FileNotFoundError(path)
            self.templates[name] = tpl
        for name in RAID_ASSETS:
            path = os.path.join(image_dir, 'rc_'+name+'.png')
            if os.path.isfile(path):
                tpl = cv2.imread(path, cv2.IMREAD_COLOR)
                if tpl is not None:
                    self.templates[name] = tpl

    def _raid_find(self, frame, name, region=None, threshold=.86):
        # Raid-specific bounded scale search; keep original Mission matching untouched.
        tpl = self.templates.get(name)
        if tpl is None or frame is None or frame.ndim != 3:
            return None
        h,w = frame.shape[:2]
        x0,y0,x1,y1 = region or RAID_REGIONS.get(name, REGIONS.get(name))
        xa,ya,xb,yb = int(x0*w),int(y0*h),int(x1*w),int(y1*h)
        crop=frame[ya:yb,xa:xb,:3]
        best=None
        for scale in (1., .975, 1.025, .95, 1.05, .925, 1.075, .90, 1.10, .875, 1.125, .85, 1.15, .825, 1.175, .80, 1.20):
            t = tpl if scale==1. else cv2.resize(tpl,None,fx=scale,fy=scale,
                interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
            th,tw=t.shape[:2]
            if crop.shape[0]<th or crop.shape[1]<tw:continue
            _,score,_,(x,y)=cv2.minMaxLoc(cv2.matchTemplate(crop,t,cv2.TM_CCOEFF_NORMED))
            if score>=threshold and (best is None or score>best.confidence):
                best=Match(name,xa+x+tw//2,ya+y+th//2,float(score),tw,th)
            if scale==1. and score>=.97:break
        return best

    @staticmethod
    def _female_border(frame, label):
        # The supplied Attack card has a solid gold outline only when selected.
        # Check both horizontal edges of the Female card, relative to its label.
        h,w=frame.shape[:2]
        scale=label.width/183.
        x0=max(0,round(label.x-115*scale)); x1=min(w,round(label.x+115*scale))
        if x1-x0<100:return False
        def edge(offset):
            y=round(label.y+offset*scale)
            pad=max(4,round(7*scale))
            strip=frame[max(0,y-pad):min(h,y+pad+1),x0:x1,:3].astype(float)
            if not strip.size:return False
            b,g,r=strip[:,:,0],strip[:,:,1],strip[:,:,2]
            gold=(r>140)&(g>90)&(r>b*1.5)&(g>b*1.3)
            return bool(np.max(gold.mean(axis=1))>=.85)
        return edge(-16) and edge(102)

    def find(self, frame: np.ndarray, name: str,
             region: Optional[tuple[float,float,float,float]]=None,
             threshold: Optional[float]=None) -> Optional[Match]:
        if frame is None or frame.ndim != 3:
            return None
        if name in RAID_REGIONS:
            return self._raid_find(frame,name,region,threshold if threshold is not None else .86)
        if name=='female_selected':
            label=self._raid_find(frame,'female_raid')
            if label and not self._raid_find(frame,'attack_selected') and self._female_border(frame,label):
                return dataclasses.replace(label,name=name)
            return None
        if name in ('raid_create_attack','raid_create_female'):
            gate='attack_selected' if name.endswith('attack') else 'female_selected'
            if self.find(frame,gate) is None:return None
            return self._raid_find(frame,'create')
        if name=='raid_start':
            return self._raid_find(frame,'start',threshold=.88)
        h,w=frame.shape[:2]
        x0,y0,x1,y1=region or REGIONS.get(name,(0,0,1,1))
        xa,ya=max(0,int(x0*w)),max(0,int(y0*h))
        xb,yb=min(w,int(x1*w)),min(h,int(y1*h))
        image=frame[ya:yb,xa:xb]
        tpl=self.templates[name]
        if image.shape[0]<tpl.shape[0] or image.shape[1]<tpl.shape[1]:
            return None
        # New input screenshots are resized to the same 1080p reference by
        # the macro's capture engine. Do not silently relax confidence.
        res=cv2.matchTemplate(image,tpl,cv2.TM_CCOEFF_NORMED)
        _lo,score,_loc,point=cv2.minMaxLoc(res)
        limit = threshold if threshold is not None else THRESHOLDS.get(name,.85)
        tw,th=tpl.shape[1],tpl.shape[0]
        result = Match(name,xa+point[0]+tw//2,ya+point[1]+th//2,score,tw,th)
        if score >= limit:
            return result  # authoritative original OpenCV path
        if name == 'reconnect' and region is None and threshold is None:
            scaled = find_scaled_reconnect(frame, tpl)
            if scaled is not None:
                return scaled
        if self.advisor is None:
            return None
        # AI only assesses the original ROI-constrained candidate. The
        # reconnect state machine still requires stable repeat observations.
        candidate=(result.x,result.y,score,tw,th)
        decision=self.advisor.verify(frame,'rc_'+name,candidate,limit,
                                      roi=(xa,ya,xb,yb))
        return dataclasses.replace(result,source='hybrid_ai') if decision.accepted else None

    def slot_a_select(self,frame):
        """Return SELECT only if geometrically inside and below the SLOT A card."""
        anchor=self.find(frame,'slot_a')
        if anchor is None:
            return None
        h,w=frame.shape[:2]
        # SELECT in the same slot column, below anchor: never B/C.
        region=(max(0,(anchor.x-120)/w), min(1,(anchor.y+80)/h),
                min(1,(anchor.x+120)/w), min(1,(anchor.y+380)/h))
        selected=self.find(frame,'select',region=region)
        return selected if selected and abs(selected.x-anchor.x)<120 else None


@dataclasses.dataclass(frozen=True)
class Action:
    name: str
    target: str
    next_marker: str
    timeout: float = 50


class ImageReconnectFlow:
    """Run a *verified* UI route. Only accepts advancement after next marker.

    capture: returns latest BGR screen frame in the application's 1080p space
    click: converts a Match into physical on-screen click (provided by main.py)
    check: throws on Stop; waits on Pause; must be called frequently
    disconnect: raises NewDisconnect if the disconnect page reappears mid-route
    progress: status message for user UI; must never include PS Code
    enter_code: types private PS code, without logging it
    mission_ready: checks the existing skill/HUD marker to confirm gameplay
    """
    def __init__(self, matcher, capture, click, check, disconnect, progress,
                 enter_code, mission_ready, server_mode='public', ps_code='', destination='mission'):
        self.matcher=matcher;self.capture=capture;self.click=click
        self.check=check;self.disconnect=disconnect;self.progress=progress
        self.enter_code=enter_code;self.mission_ready=mission_ready
        self.mode=server_mode;self.ps_code=ps_code
        if destination not in ('mission','attack','female'):
            raise ValueError('Unsupported reconnect destination')
        self.destination=destination
        if self.mode not in ('public','ps_code','gamepass'):
            raise ValueError('Unsupported server mode')
        if self.mode=='ps_code' and (not ps_code or any(ord(c)<32 for c in ps_code) or len(ps_code)>128):
            raise ValueError('Valid PS Code required')

    def _sleep(self, seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            self.check()
            time.sleep(min(.12, max(0,until-time.monotonic())))

    def _wait_for(self,name,timeout=55,quiet=False,stable=2,allow_disconnect=False):
        expiry=time.monotonic()+timeout; count=0;last=None
        while time.monotonic()<expiry:
            self.check()
            frame=self.capture()
            if not allow_disconnect and self.disconnect(frame):
                raise NewDisconnect('Reconnect appeared again during navigation')
            match=(self.matcher.slot_a_select(frame) if name=='slot_select'
                   else self.matcher.find(frame,name))
            if match is not None and last is not None and abs(last.x-match.x)<16 and abs(last.y-match.y)<16:
                count+=1
            else:count=1 if match else 0
            last=match
            if match and count>=stable:
                return match
            self._sleep(.3)
        return None

    def _step(self,index,total,name,target,next_marker,timeout=50):
        """Retry one action only when the same target remains visible.
        A button being clicked is not success. Next-screen marker is success.
        """
        for attempt in range(1,4):
            self.progress(index,total,name,'Detecting',attempt)
            hit=self._wait_for(target,timeout=timeout)
            if hit is None:
                raise ReconnectError(f'{name}: {target} not detected')
            # Verify *fresh* screenshot immediately before interacting.
            fresh=self.capture()
            if self.disconnect(fresh):raise NewDisconnect('Reconnect returned')
            still=(self.matcher.slot_a_select(fresh) if target=='slot_select'
                   else self.matcher.find(fresh,target))
            if not still or abs(still.x-hit.x)>16 or abs(still.y-hit.y)>16:
                continue
            self.progress(index,total,name,'Clicking',attempt)
            self.check();self.click(still)
            if name=='Enter PS Code':
                self.check();self.enter_code(self.ps_code)
            self.progress(index,total,name,'Verifying',attempt)
            if name=='Enter PS Code':
                # GENERATE was already visible before typing the PS Code.
                # Its continued presence is NOT evidence the code was entered.
                # Verify the original three-dash placeholder disappeared,
                # and that Generate is still in the VIP Servers panel.
                expiry=time.monotonic()+min(15,timeout)
                verified=0
                while time.monotonic()<expiry:
                    self.check()
                    frame=self.capture()
                    if self.disconnect(frame):raise NewDisconnect('Reconnect returned')
                    if (self.matcher.find(frame,'ps_input') is None and
                            self.matcher.find(frame,'generate') is not None):
                        verified+=1
                    else:verified=0
                    if verified>=2:break
                    self._sleep(.3)
                next_confirmed=verified>=2
            else:
                next_confirmed=bool(self._wait_for(next_marker, timeout=timeout))
            if next_confirmed:
                self.progress(index,total,name,'Confirmed',attempt)
                return True
            self.progress(index,total,name,'Retrying',attempt)
        raise ReconnectError(f'{name}: next screen {next_marker} not confirmed')

    def _ensure_vip_servers(self,index,total):
        """Do not click CODES if VIP SERVERS is already open in the menu.

        The user's full screenshot shows a separate CODES/REDEEM panel and
        VIP SERVERS controls simultaneously. Their positions must not be
        mistaken for a screen transition caused by clicking CODES.
        """
        self.progress(index,total,'Locate VIP Servers','Detecting',1)
        if self._wait_for('generate',timeout=2,stable=2):
            self.progress(index,total,'Locate VIP Servers','Confirmed',1)
            return
        # If game UI changes and VIP Servers is hidden, a CODES click may
        # expose it; verify the VIP-only GENERATE control appears afterwards.
        self._step(index,total,'Open VIP Servers','codes','generate',timeout=15)

    def run(self):
        route=[('Select Slot A','slot_select','play_small')]
        if self.mode=='public':
            route += [('Open Public Servers','play_small','lobby'),
                      ('Join Lobby','lobby','play_large')]
        elif self.mode=='ps_code':
            route += [('Ensure VIP Servers','codes','generate'),
                      ('Enter PS Code','ps_input','generate'),
                      # Generate returns to the LOBBY / TRADE choice screen.
                      # Do not wait for the large PLAY button until LOBBY is clicked.
                      ('Generate Private Server','generate','lobby'),
                      ('Join Lobby','lobby','play_large')]
        else:
            # Preserve the original Gamepass path until its post-Generate
            # screen has been verified; the reported bug concerns PS Code.
            route += [('Ensure VIP Servers','codes','generate'),
                      ('Generate Gamepass Server','generate','play_large')]
        raid=self.destination!='mission'
        start_marker='raid_start' if raid else 'start'
        if raid:
            route += [('Open Raid Menu','play_large','raids'),
                      ('Select Raids','raids','attack_selected')]
            if self.destination=='female':
                route += [('Select Female Titan','female_raid','female_selected')]
            route += [('Create Raid','raid_create_'+self.destination,start_marker)]
        else:
            route += [('Open Missions Menu','play_large','missions'),
                      ('Select Missions','missions','outskirts'),
                      # CREATE can already be visible with SHIGANSHINA selected.
                      ('Select Outskirts','outskirts','outskirts_selected'),
                      ('Create Mission','create','start')]
        total=len(route)+1
        for n,(label,target,gate) in enumerate(route,1):
            if label=='Ensure VIP Servers':
                self._ensure_vip_servers(n,total)
            else:
                self._step(n,total,label,target,gate,timeout=75 if (target=='generate' or gate=='play_large') else 40)
        # START must be the modal's START, not the button behind the modal.
        kind='Raid' if raid else 'Mission'
        self.progress(total,total,'Start '+kind,'Detecting',1)
        button=self._wait_for(start_marker,timeout=35)
        if not button:raise ReconnectError(kind+' START is not visible in the room popup')
        if raid:
            fresh=self.capture()
            if self.disconnect(fresh):raise NewDisconnect('Reconnect returned before START')
            still=self.matcher.find(fresh,start_marker)
            if not still or abs(still.x-button.x)>16 or abs(still.y-button.y)>16:
                raise ReconnectError('Raid START moved before clicking')
            button=still
        self.check();self.click(button)
        self.progress(total,total,'Waiting for '+kind+' HUD','Verifying',1)
        expiry=time.monotonic()+100
        while time.monotonic()<expiry:
            self.check();frame=self.capture()
            if self.disconnect(frame):raise NewDisconnect('Disconnected while mission was loading')
            if self.mission_ready(frame):
                self.progress(total,total,kind+' Confirmed','Confirmed',1)
                return True
            self._sleep(.4)
        raise ReconnectError(kind+' HUD not confirmed after START')

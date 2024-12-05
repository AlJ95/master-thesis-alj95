import json

dataset_dict = {
    "test_cases": [
        # TestCases Mieter
        {
            "input": "Muss ich Nachteile in Kauf nehmen, weil die Wärmezähler zu spät und falsch eingebaut wurden, und hätte der Verbrauch ab 11/2023 verbrauchsgerecht abgerechnet werden müssen?",        
            "actual_output": "",
            "expected_output": "Gemäß § 12 Abs. 1 Satz 1 der Heizkostenverordnung (HKVO) hast du das Recht, die Heizkosten um 15 % zu kürzen, wenn die Wärmezähler nicht ordnungsgemäß installiert wurden. Dies berücksichtigt das Verursacherprinzip und ermöglicht eine Reduzierung der Heizkosten aufgrund der verspäteten und fehlerhaften Installation der Zähler.",
            "retrieval_context": [
                "§ 12 Abs. 1 Satz 1 Heizkostenverordnung (HKVO): (1) Soweit die Kosten der Versorgung mit Wärme oder Warmwasser entgegen den Vorschriften dieser Verordnung nicht verbrauchsabhängig abgerechnet werden, hat der Nutzer das Recht, bei der nicht verbrauchsabhängigen Abrechnung der Kosten den auf ihn entfallenden Anteil um 15 vom Hundert zu kürzen. Wenn der Gebäudeeigentümer entgegen § 5 Absatz 2 oder Absatz 3 keine fernablesbare Ausstattung zur Verbrauchserfassung installiert hat, hat der Nutzer das Recht, bei der Abrechnung der Kosten den auf ihn entfallenden Anteil um 3 vom Hundert zu kürzen.",
            ]
        },
        {
            "input": "Darf der Vermieter die Kosten für die Fensterwartung auf den Mieter umlegen, auch wenn dies nicht explizit im Mietvertrag vereinbart wurde?",
            "actual_output": "",
            "expected_output": "Laut § 2 BetrKV Punkt 17 können sonstige Betriebskosten nur umgelegt werden, wenn sie explizit im Mietvertrag genannt sind. Ein allgemeiner Verweis auf die Betriebskostenverordnung reicht nicht aus. Das Amtsgericht Rheine (AG Rheine) entschied am 10.10.2018 (10 C 112/18), dass Fensterwartungskosten explizit vereinbart werden müssen, um umgelegt werden zu können. Fehlt eine solche Vereinbarung, ist die Umlage unzulässig.",
            "retrieval_context": [
                "$ 2 BetrKV Verordnung über die Aufstellung von Betriebskosten (Betriebskostenverordnung - BetrKV) Absatz 1: Betriebskosten sind die Kosten, die dem Eigentümer oder Erbbauberechtigten durch das Eigentum oder Erbbaurecht am Grundstück oder durch den bestimmungsmäßigen Gebrauch des Gebäudes, der Nebengebäude, Anlagen, Einrichtungen und des Grundstücks laufend entstehen. Sach- und Arbeitsleistungen des Eigentümers oder Erbbauberechtigten dürfen mit dem Betrag angesetzt werden, der für eine gleichwertige Leistung eines Dritten, insbesondere eines Unternehmers, angesetzt werden könnte; die Umsatzsteuer des Dritten darf nicht angesetzt werden.",
                """$ 2 BetrKV Verordnung über die Aufstellung von Betriebskosten (Betriebskostenverordnung - BetrKV) Absatz 2:  Zu den Betriebskosten gehören nicht:
1.
die Kosten der zur Verwaltung des Gebäudes erforderlichen Arbeitskräfte und Einrichtungen, die Kosten der Aufsicht, der Wert der vom Vermieter persönlich geleisteten Verwaltungsarbeit, die Kosten für die gesetzlichen oder freiwilligen Prüfungen des Jahresabschlusses und die Kosten für die Geschäftsführung (Verwaltungskosten),
2.
die Kosten, die während der Nutzungsdauer zur Erhaltung des bestimmungsmäßigen Gebrauchs aufgewendet werden müssen, um die durch Abnutzung, Alterung und Witterungseinwirkung entstehenden baulichen oder sonstigen Mängel ordnungsgemäß zu beseitigen (Instandhaltungs- und Instandsetzungskosten)."""
                "§ 2 BetrKV Punkt 17: sonstige Betriebskosten, hierzu gehören Betriebskosten im Sinne des § 1, die von den Nummern 1 bis 16 nicht erfasst sind."
            ]
        },
        {
            "input": "Handelt es sich bei dem Austausch einer defekten Heizungsanlage gegen eine neue, gleichwertige Anlage um eine Instandsetzung oder eine Modernisierung, und dürfen die Kosten auf die Mieter umgelegt werden?",
            "actual_output": "",
            "expected_output": "Laut § 555b BGB handelt es sich um eine Modernisierungsmaßnahme, wenn durch den Austausch der Heizungsanlage Endenergie nachhaltig eingespart wird. Wenn jedoch eine alte Anlage durch eine gleichwertige ersetzt wird, handelt es sich um eine Erhaltungsmaßnahme (Instandhaltung oder Instandsetzung), und die Kosten dürfen nicht auf die Mieter umgelegt werden. Der Vermieter muss die Modernisierungskosten genau vorrechnen und die notwendigen Maßnahmen zur Reparatur der alten Anlage berücksichtigen.",
            "retrieval_context": [
                """§ 555b Modernisierungsmaßnahmen: Modernisierungsmaßnahmen sind bauliche Veränderungen,
1.
durch die in Bezug auf die Mietsache Endenergie nachhaltig eingespart wird (energetische Modernisierung),
1a.
durch die mittels Einbaus oder Aufstellung einer Heizungsanlage zum Zwecke der Inbetriebnahme in einem Gebäude die Anforderungen des § 71 des Gebäudeenergiegesetzes erfüllt werden,
2.
durch die nicht erneuerbare Primärenergie nachhaltig eingespart oder das Klima nachhaltig geschützt wird, sofern nicht bereits eine energetische Modernisierung nach Nummer 1 vorliegt,
3.
durch die der Wasserverbrauch nachhaltig reduziert wird,
4.
durch die der Gebrauchswert der Mietsache nachhaltig erhöht wird,
4a.
durch die die Mietsache erstmalig mittels Glasfaser an ein öffentliches Netz mit sehr hoher Kapazität im Sinne des § 3 Nummer 33 des Telekommunikationsgesetzes angeschlossen wird,
5.
durch die die allgemeinen Wohnverhältnisse auf Dauer verbessert werden,
6.
die auf Grund von Umständen durchgeführt werden, die der Vermieter nicht zu vertreten hat, und die keine Erhaltungsmaßnahmen nach § 555a sind, oder
7.
durch die neuer Wohnraum geschaffen wird."""
            ]
        },
        {
            "input": "Ist eine Mietminderung gerechtfertigt, wenn die vom Vermieter gestellte Waschmaschine seit 2 Monaten kaputt ist und der Vermieter keinen Handwerker beauftragt hat, um sie zu reparieren oder zu ersetzen?",
            "actual_output": "",
            "expected_output": "Laut § 536 BGB ist eine Mietminderung gerechtfertigt, wenn die Mietsache einen Mangel aufweist, der ihre Tauglichkeit zum vertragsgemäßen Gebrauch mindert. Da die Waschmaschine im Mietvertrag als Bestandteil der Mietsache aufgeführt ist und seit 2 Monaten nicht funktioniert, ist eine Mietminderung für diesen Zeitraum gerechtfertigt. Der Vermieter ist verpflichtet, die Mietsache in einem vertragsgemäßen Zustand zu erhalten.",
            "retrieval_context": [
                "$ 536 BGB Mietminderung bei Sach- und Rechtsmängeln: (1) Hat die Mietsache zur Zeit der Überlassung an den Mieter einen Mangel, der ihre Tauglichkeit zum vertragsgemäßen Gebrauch aufhebt, oder entsteht während der Mietzeit ein solcher Mangel, so ist der Mieter für die Zeit, in der die Tauglichkeit aufgehoben ist, von der Entrichtung der Miete befreit. Für die Zeit, während der die Tauglichkeit gemindert ist, hat er nur eine angemessen herabgesetzte Miete zu entrichten. Eine unerhebliche Minderung der Tauglichkeit bleibt außer Betracht.",
                "$ 536 BGB Mietminderung bei Sach- und Rechtsmängeln: (1a) Für die Dauer von drei Monaten bleibt eine Minderung der Tauglichkeit außer Betracht, soweit diese auf Grund einer Maßnahme eintritt, die einer energetischen Modernisierung nach § 555b Nummer 1 dient.",
                "$ 536 BGB Mietminderung bei Sach- und Rechtsmängeln: (2) Absatz 1 Satz 1 und 2 gilt auch, wenn eine zugesicherte Eigenschaft fehlt oder später wegfällt.",
                "$ 536 BGB Mietminderung bei Sach- und Rechtsmängeln: (3) Wird dem Mieter der vertragsgemäße Gebrauch der Mietsache durch das Recht eines Dritten ganz oder zum Teil entzogen, so gelten die Absätze 1 und 2 entsprechend.",
                "$ 536 BGB Mietminderung bei Sach- und Rechtsmängeln: (4) Bei einem Mietverhältnis über Wohnraum ist eine zum Nachteil des Mieters abweichende Vereinbarung unwirksam."
            ]
        },
        {
            "input": "Wer muss für die Stromkostendifferenz aufkommen, wenn der Mieter aufgrund eines Zwischenzählers den Strom über den Hauptzähler des Nachbarn abrechnen muss, obwohl im Mietvertrag steht, dass der Strom direkt über den Versorger abgerechnet werden soll?",
            "actual_output": "",
            "expected_output": "Der Vermieter muss den Strompreis ohne Auf- oder Abschlag an den Mieter weiterreichen, da der Mieter Vertragspartner des Vermieters ist. Der Vermieter ist verpflichtet, die technischen Voraussetzungen zu schaffen, damit der Mieter seinen Strom direkt beim Versorger abrechnen kann. Dies ergibt sich aus dem Messstellenbetriebsgesetz und der freien Lieferantenwahl in Deutschland. (§ 675 Abs. 2 BGB, Messstellenbetriebsgesetz)",
            "retrieval_context": [
                "$ 675 Abs. 2 BGB: (2) Wer einem anderen einen Rat oder eine Empfehlung erteilt, ist, unbeschadet der sich aus einem Vertragsverhältnis, einer unerlaubten Handlung oder einer sonstigen gesetzlichen Bestimmung ergebenden Verantwortlichkeit, zum Ersatz des aus der Befolgung des Rates oder der Empfehlung entstehenden Schadens nicht verpflichtet."
            ]
        },
        {
            "input": "Wie lange hat der Vermieter Zeit, um auf eine Anfechtung der Heizkostenabrechnung zu reagieren?",
            "actual_output": "",
            "expected_output": "Es gibt keine festgelegte Frist im Gesetz, die den Vermieter verpflichtet, innerhalb einer bestimmten Zeit auf eine Anfechtung zu reagieren. Es handelt sich um eine Einzelfallentscheidung. Der Mieter sollte jedoch sicherstellen, dass er seine Einwände klar und nachvollziehbar darlegt und gegebenenfalls Akteneinsicht verlangt, um die Berechnungsgrundlagen zu überprüfen. Ein Verweis auf § 675 Abs. 2 BGB könnte hilfreich sein, um die Pflichten des Vermieters zu verdeutlichen.",
            "retrieval_context": [
                "$ 675 Abs. 2 BGB: (2) Wer einem anderen einen Rat oder eine Empfehlung erteilt, ist, unbeschadet der sich aus einem Vertragsverhältnis, einer unerlaubten Handlung oder einer sonstigen gesetzlichen Bestimmung ergebenden Verantwortlichkeit, zum Ersatz des aus der Befolgung des Rates oder der Empfehlung entstehenden Schadens nicht verpflichtet."
            ]
        },
        {
            "input": " Kann ein Mieter einen befristeten Mietvertrag vorzeitig kündigen, wenn die im Vertrag genannten baulichen Maßnahmen nicht konkret terminiert oder vorbereitet sind?",
            "actual_output": "",
            "expected_output": "Laut § 575 BGB muss die Begründung für die Befristung so umfassend sein, dass der Mieter erkennen kann, dass seine Wohnung während der geplanten Maßnahmen nicht bewohnbar ist. Wenn die Begründung unklar ist oder keine konkreten Maßnahmen und Termine genannt werden, kann die Befristung unwirksam sein und der Mietvertrag kann jederzeit mit gesetzlicher Frist gekündigt werden. Ein Gerichtsurteil (AG Tempelhof-Kreuzberg vom 25.02.2019, Az.: 7 C 285/18) bestätigt, dass unklare Begründungen zur Unwirksamkeit der Befristung führen können.",
            "retrieval_context": [
                """§ 575 Zeitmietvertrag BGB (1): Ein Mietverhältnis kann auf bestimmte Zeit eingegangen werden, wenn der Vermieter nach Ablauf der Mietzeit
1.
die Räume als Wohnung für sich, seine Familienangehörigen oder Angehörige seines Haushalts nutzen will,
2.
in zulässiger Weise die Räume beseitigen oder so wesentlich verändern oder instand setzen will, dass die Maßnahmen durch eine Fortsetzung des Mietverhältnisses erheblich erschwert würden, oder
3.
die Räume an einen zur Dienstleistung Verpflichteten vermieten will
und er dem Mieter den Grund der Befristung bei Vertragsschluss schriftlich mitteilt. Anderenfalls gilt das Mietverhältnis als auf unbestimmte Zeit abgeschlossen.""",
                "§ 575 Zeitmietvertrag BGB (2): Der Mieter kann vom Vermieter frühestens vier Monate vor Ablauf der Befristung verlangen, dass dieser ihm binnen eines Monats mitteilt, ob der Befristungsgrund noch besteht. Erfolgt die Mitteilung später, so kann der Mieter eine Verlängerung des Mietverhältnisses um den Zeitraum der Verspätung verlangen.",
                "§ 575 Zeitmietvertrag BGB (3): Tritt der Grund der Befristung erst später ein, so kann der Mieter eine Verlängerung des Mietverhältnisses um einen entsprechenden Zeitraum verlangen. Entfällt der Grund, so kann der Mieter eine Verlängerung auf unbestimmte Zeit verlangen. Die Beweislast für den Eintritt des Befristungsgrundes und die Dauer der Verzögerung trifft den Vermieter.",
                "§ 575 Zeitmietvertrag BGB (4): Eine zum Nachteil des Mieters abweichende Vereinbarung ist unwirksam."
            ]
        },
        {
            "input": "Was kann ich tun, wenn die Farbe von meinem Balkon herunterfällt und die Vermieterin trotz mehrfacher Aufforderung das Problem nicht behebt?",
            "actual_output": "",
            "expected_output": "Der Mieter kann gemäß § 536 BGB eine Mietminderung verlangen, wenn der Gebrauch der Mietsache erheblich beeinträchtigt ist. Zudem kann der Mieter eine Frist zur Mängelbeseitigung setzen und bei Fristablauf Schadensersatzansprüche geltend machen. Es ist ratsam, die Vermieterin schriftlich aufzufordern, den Mangel innerhalb einer angemessenen Frist zu beheben.",
            "retrieval_context": [
                ""
            ]
        },
        {
            "input": "Kann der eigentliche Besitzer einer Garage, die ich bisher ohne Mietvertrag genutzt habe, rückwirkend Miete von mir verlangen?",
            "actual_output": "",
            "expected_output": "Es könnte eine Nutzungsentschädigung in Betracht kommen, da die Garage ohne Gegenleistung genutzt wurde. Dies könnte unter Umständen als Schadensersatzanspruch geltend gemacht werden, abhängig von den genauen Umständen und der rechtlichen Bewertung der Nutzung. (§ 675 Abs. 2 BGB wurde erwähnt).",
            "retrieval_context": [
                "§675 BGB (2): Wer einem anderen einen Rat oder eine Empfehlung erteilt, ist, unbeschadet der sich aus einem Vertragsverhältnis, einer unerlaubten Handlung oder einer sonstigen gesetzlichen Bestimmung ergebenden Verantwortlichkeit, zum Ersatz des aus der Befolgung des Rates oder der Empfehlung entstehenden Schadens nicht verpflichtet."
            ]
        },
        {
            "input": "Wer ist verantwortlich für die Beseitigung und die Kosten eines Schimmelflecks in einer Waschküche, die sich in einem separaten Gebäude auf dem Hof befindet und keine Heizung hat?",
            "actual_output": "",
            "expected_output": "Der Vermieter könnte verantwortlich sein, da der Raum keine Heizung hat und somit möglicherweise ungeeignet für die Nutzung als Waschküche ist. Dies könnte als baulicher Mangel betrachtet werden. Allerdings müsste der Mieter nachweisen, dass der Schimmel nicht durch unsachgemäße Nutzung entstanden ist. Laut § 675 Abs. 2 BGB haftet der Mieter nicht für Schäden, für die er nicht verantwortlich ist.",
            "retrieval_context": [
                "§675 BGB (2): Wer einem anderen einen Rat oder eine Empfehlung erteilt, ist, unbeschadet der sich aus einem Vertragsverhältnis, einer unerlaubten Handlung oder einer sonstigen gesetzlichen Bestimmung ergebenden Verantwortlichkeit, zum Ersatz des aus der Befolgung des Rates oder der Empfehlung entstehenden Schadens nicht verpflichtet."
            ]
        },
        # TestCases Vermieter
        {
            "input": "Darf der Vermieter ohne Zustimmung des Mieters den Garten eines vermieteten Einfamilienhauses betreten, um notwendige Gartenarbeiten durchzuführen, wenn der Mieter seinen Pflichten zur Gartenpflege nicht nachkommt?",
            "actual_output": "",
            "expected_output": "Laut § 555a BGB muss der Mieter notwendige Erhaltungsmaßnahmen dulden, die rechtzeitig angekündigt werden müssen, es sei denn, sie sind nur mit einer unerheblichen Einwirkung auf die Mietsache verbunden oder ihre sofortige Durchführung ist zwingend erforderlich. Der Vermieter muss jedoch die Zustimmung des Mieters einholen, um den Garten zu betreten, und kann nicht ohne Weiteres eigenmächtig handeln",
            "retrieval_context": [
                "§ 555a BGB Erhaltungsmaßnahmen (1): Der Mieter hat Maßnahmen zu dulden, die zur Instandhaltung oder Instandsetzung der Mietsache erforderlich sind (Erhaltungsmaßnahmen).",
                "§ 555a BGB Erhaltungsmaßnahmen (2): Erhaltungsmaßnahmen sind dem Mieter rechtzeitig anzukündigen, es sei denn, sie sind nur mit einer unerheblichen Einwirkung auf die Mietsache verbunden oder ihre sofortige Durchführung ist zwingend erforderlich.",
                "§ 555a BGB Erhaltungsmaßnahmen (3): Aufwendungen, die der Mieter infolge einer Erhaltungsmaßnahme machen muss, hat der Vermieter in angemessenem Umfang zu ersetzen. Auf Verlangen hat er Vorschuss zu leisten.",
                "§ 555a BGB Erhaltungsmaßnahmen (4): Eine zum Nachteil des Mieters von Absatz 2 oder 3 abweichende Vereinbarung ist unwirksam."
            ]
        },
        {
            "input": "Kann ich als Vermieter das Mietverhältnis fristlos kündigen, wenn der Mieter mit der Mietzahlung in Verzug gerät?", 
            "actual_output": "", 
            "expected_output": "Ein Vermieter kann das Mietverhältnis fristlos kündigen, wenn der Mieter mit der Mietzahlung in Verzug gerät. Allerdings muss der Vermieter dem Mieter zuerst eine angemessene Frist zur Zahlung setzen (§ 543 Abs. 3 BGB). Wenn der Mieter innerhalb dieser Frist nicht zahlt, kann der Vermieter das Mietverhältnis fristlos kündigen (§ 543 Abs. 1 BGB).",
            "retrieval_context": [  
                "§ 543 Abs. 1 BGB: Fristlose Kündigung aus wichtigem Grund. Ein Mietverhältnis kann von beiden Parteien ohne Einhaltung einer Kündigungsfrist gekündigt werden, wenn ein wichtiger Grund vorliegt.",
                "§ 543 Abs. 2 Nr. 3 BGB: Ein wichtiger Grund zur fristlosen Kündigung liegt insbesondere vor, wenn der Mieter in zwei aufeinanderfolgenden Monaten mit der Entrichtung der Miete oder eines nicht unerheblichen Teils der Miete in Verzug ist oder wenn der Mieter in einem Zeitraum, der sich über mehr als zwei Termine erstreckt, mit der Entrichtung der Miete in Höhe eines Betrages in Verzug ist, der die Miete für zwei Monate erreicht.",
                "§ 543 Abs. 3 BGB: Vor der fristlosen Kündigung muss der Vermieter dem Mieter eine angemessene Frist zur Zahlung setzen, es sei denn, eine solche Fristsetzung ist aufgrund besonderer Umstände des Einzelfalls entbehrlich. Wenn der Mieter innerhalb dieser Frist nicht zahlt, kann der Vermieter das Mietverhältnis fristlos kündigen."
                ] 
        },
        {
            "input": "Wer ist für die Durchführung von Schönheitsreparaturen verantwortlich?",
            "actual_output": "", 
            "expected_output": "Grundsätzlich ist der Vermieter für die Durchführung von Schönheitsreparaturen verantwortlich (§ 535 Abs. 1 Satz 2 BGB). Diese Pflicht kann jedoch im Mietvertrag auf den Mieter übertragen werden, sofern die entsprechende Klausel klar und verständlich formuliert ist. Unklare oder zu weitgehende Klauseln sind unwirksam (§ 307 BGB). Schönheitsreparaturen umfassen Maßnahmen, die den ursprünglichen Zustand der Mietwohnung wiederherstellen oder erhalten sollen, wie das Streichen, Tapezieren und das Lackieren von Heizkörpern, Türen und Fenstern von innen. Der Mieter darf die Wohnung während der Mietzeit nach seinen Vorstellungen gestalten, muss sie jedoch bei Auszug in einem neutralen Zustand zurückgeben. Laut Rechtsprechung des Bundesgerichtshofs (BGH) müssen extreme oder sehr dunkle Farben vor dem Auszug beseitigt werden, um eine problemlose Weitervermietung zu ermöglichen (BGH, Urteil vom 6. November 2013 – VIII ZR 416/12). Starre Fristen für die Durchführung von Schönheitsreparaturen sind unwirksam (BGH, Urteil vom 23. Juni 2004 – VIII ZR 361/03). Zusammengefasst ist der Vermieter für Schönheitsreparaturen verantwortlich, sofern nicht eine wirksame vertragliche Regelung diese Pflicht auf den Mieter überträgt.",
            "retrieval_context": [
                "§ 535 Abs. 1 Satz 2 BGB: Der Vermieter ist verpflichtet, die Mietsache in einem zum vertragsgemäßen Gebrauch geeigneten Zustand zu überlassen und während der Mietzeit zu erhalten. Dies umfasst auch die Durchführung von Schönheitsreparaturen.",
                "§ 307 BGB: Inhaltskontrolle von Allgemeinen Geschäftsbedingungen. Eine Klausel im Mietvertrag, die die Schönheitsreparaturen auf den Mieter überträgt, darf den Mieter nicht unangemessen benachteiligen. Unklare oder zu weitgehende Klauseln sind unwirksam.",
                "BGH, Urteil vom 6. November 2013 – VIII ZR 416/12: Klauseln, die den Mieter verpflichten, extreme oder sehr dunkle Farben vor dem Auszug zu beseitigen, sind zulässig, um eine problemlose Weitervermietung zu ermöglichen.",
                "BGH, Urteil vom 23. Juni 2004 – VIII ZR 361/03: Starre Fristen für die Durchführung von Schönheitsreparaturen sind unwirksam. Klauseln müssen den tatsächlichen Zustand der Wohnung berücksichtigen."
                ] 
        },
        {
            "input": "Kann ich als Vermieter eine Wohnung gegen den Willen des Mieters räumen lassen?",
            "actual_output": "", 
            "expected_output": "Nein, als Vermieter können Sie eine Wohnung nicht gegen den Willen des Mieters räumen lassen, ohne ein gerichtliches Verfahren einzuleiten. Eine Wohnung kann nur nach einem Räumungsprozess aufgrund eines Räumungsurteils oder Räumungsvergleichs zwangsgeräumt werden. Ein Räumungsurteil setzt mindestens eine Räumungsklage voraus (§ 546 Abs. 1 BGB), die wiederum erfordert, dass dem Mieter wirksam die Kündigung erklärt wurde und eine etwaige Kündigungsfrist abgelaufen ist (§ 573 BGB). Die Polizei räumt im Auftrag des Vermieters keine Wohnungen. Wenn der Vermieter selbst oder mit eigenen Helfern versucht, die Wohnung gegen den Willen des Mieters zu räumen, macht er sich strafbar (§ 123 StGB - Hausfriedensbruch; § 240 StGB - Nötigung). Zusammengefasst, eine Zwangsräumung darf nur durch ein gerichtliches Verfahren erfolgen, wobei ein Räumungstitel durch eine Räumungsklage und ein entsprechendes Urteil erlangt werden muss. Eigenmächtiges Handeln oder Selbsthilfe ist rechtlich nicht zulässig.",
            "retrieval_context": [ 
                "§ 546 Abs. 1 BGB: Nach Beendigung des Mietverhältnisses ist der Mieter verpflichtet, die Mietsache an den Vermieter zurückzugeben.",
                "§ 573 BGB: Ordentliche Kündigung des Mietverhältnisses durch den Vermieter. Eine Kündigung ist nur unter bestimmten Voraussetzungen zulässig, und der Vermieter muss die Kündigungsfristen einhalten.",
                "§ 123 StGB: Hausfriedensbruch. Der Vermieter macht sich strafbar, wenn er gegen den Willen des Mieters dessen Wohnung betritt oder räumt.",
                "§ 240 StGB: Nötigung. Der Vermieter macht sich strafbar, wenn er versucht, den Mieter durch Gewalt oder Drohung zur Räumung der Wohnung zu zwingen."
                ]
        },
        {
            "input": "Darf ich als Vermieter ungefragt in die Wohnung eintreten?", 
            "actual_output": "", 
            "expected_output": "Gemäß § 809 BGB muss der Mieter den Vermieter auf Anfrage in die Wohnung lassen, wenn der Vermieter einen berechtigten Grund hat, die Wohnung zu betreten. Solche Gründe können Renovierungs- und Wartungsarbeiten, Mess- und Ablesetermine sowie Gefahr oder konkreter Verdacht sein. Der Vermieter muss den Besuch rechtzeitig ankündigen und sich an ortsübliche Zeiten halten. Der Mieter kann unberechtigten Zutritt verweigern, sollte jedoch darauf achten, dass er sich nicht schadensersatzpflichtig macht. Im Falle eines Mieterwechsels oder Verkaufs der Wohnung muss der Mieter dem Vermieter die Möglichkeit zur Wohnungsbesichtigung geben.",
            "retrieval_context": [
                "§ 809 BGB: Der Besitzer einer Sache kann von demjenigen, der ein Recht an der Sache hat, verlangen, dass dieser ihm den Zutritt zu der Sache gestattet, soweit die Ausübung des Rechts dies erfordert. Dazu gehört auch das Betreten einer Mietwohnung durch den Vermieter, wenn ein berechtigter Grund vorliegt.",
                "§ 535 Abs. 1 Satz 2 BGB: Der Vermieter ist verpflichtet, die Mietsache in einem zum vertragsgemäßen Gebrauch geeigneten Zustand zu überlassen und während der Mietzeit zu erhalten. Dies kann Besuche des Vermieters zur Überprüfung des Zustands der Wohnung erforderlich machen.",
                "Rechtsprechung des Bundesgerichtshofs (BGH): Der Vermieter muss den Besuch rechtzeitig ankündigen und sich an ortsübliche Zeiten halten. Unangekündigte Besuche sind grundsätzlich unzulässig.",
                "§ 541 BGB: Wenn der Mieter seine Pflicht zur Duldung des Zutritts verweigert, kann der Vermieter eine einstweilige Verfügung zur Duldung des Zutritts erwirken."
                ]
        },
        {
            "input": "Wie kann ein Vermieter mit Mietnomaden umgehen, die spurlos verschwinden und die Wohnung beschädigt und verwahrlost zurücklassen?", 
            "actual_output": "", 
            "expected_output": "Der Vermieter sollte das Mietverhältnis ordnungsgemäß kündigen, um die Wohnung wieder vermietbar zu machen. Dazu ist eine fristlose Kündigung erforderlich, wenn der Mieter mit zwei aufeinanderfolgenden Monatsmieten im Verzug ist (§ 543 II 1 Nr. 3a BGB). Die Kündigung muss schriftlich erfolgen und dem Mieter zugehen. Wenn der Mieter nicht auffindbar ist, kann eine öffentliche Zustellung beim Amtsgericht beantragt werden. Nach Beendigung des Räumungsprozesses kann der Vermieter die Wohnung neu vermieten. Zusätzlich hat der Vermieter Ansprüche auf Zahlung des Mietzinses bis zur Beendigung des Mietverhältnisses und auf Nutzungsentschädigung danach (§ 546a BGB). Schadensersatzansprüche können bei Schäden geltend gemacht werden, die über den normalen Gebrauch hinausgehen. Es ist ratsam, einen erfahrenen Rechtsanwalt im Mietrecht hinzuzuziehen, um die rechtlichen Interessen des Vermieters zu wahren.",
            "retrieval_context": [
                "§ 543 Abs. 2 Nr. 3a BGB: Der Vermieter kann das Mietverhältnis fristlos kündigen, wenn der Mieter mit zwei aufeinander folgenden Monatsmieten im Verzug ist.",
                "§ 546a BGB: Der Mieter ist verpflichtet, die Mietsache nach Beendigung des Mietverhältnisses zurückzugeben. Für den Zeitraum bis zur Rückgabe der Mietsache ist der Mieter verpflichtet, die Miete weiter zu zahlen. Der Vermieter kann auch eine Nutzungsentschädigung verlangen, wenn der Mieter die Wohnung nach Beendigung des Mietverhältnisses weiterhin nutzt.",
                "§ 562 BGB: Der Vermieter hat Anspruch auf Zahlung der Miete bis zum Ende des Mietverhältnisses, auch wenn die Wohnung durch den Mieter beschädigt oder verwahrlost zurückgelassen wird.",
                "Rechtsprechung: Schadensersatzansprüche können geltend gemacht werden für Schäden, die über den normalen Gebrauch hinausgehen. Der Vermieter sollte auch die Möglichkeit der öffentlichen Zustellung beim Amtsgericht prüfen, wenn der Mieter nicht auffindbar ist.",
                "Empfehlung: Es ist ratsam, einen erfahrenen Rechtsanwalt im Mietrecht hinzuzuziehen, um die rechtlichen Interessen des Vermieters im Falle von Mietnomaden zu wahren."
                ] 
        },
        {
            "input": "Darf ich als Vermieter den Mietvertrag fristlos kündigen, wenn der Mieter eine falsche Selbstauskunft bezüglich Mietschulden aus früheren Mietverhältnissen abgegeben hat?", 
            "actual_output": "", 
            "expected_output": "Gemäß § 543 Abs. 2 Nr. 3 BGB kann der Vermieter das Mietverhältnis fristlos kündigen, wenn der Mieter seine vertraglichen Pflichten schuldhaft nicht unerheblich verletzt. Eine arglistige Täuschung des Vermieters durch den Mieter kann eine solche Vertragsverletzung darstellen.",
            "retrieval_context": [
                "§ 543 Abs. 2 Nr. 3 BGB: Der Vermieter kann das Mietverhältnis fristlos kündigen, wenn der Mieter seine vertraglichen Pflichten schuldhaft nicht unerheblich verletzt. Eine solche Vertragsverletzung liegt vor, wenn der Mieter wesentliche Vertragspflichten verletzt, z.B. durch arglistige Täuschung.",
                "Rechtsprechung: Arglistige Täuschung des Vermieters durch falsche Angaben in der Selbstauskunft kann als schwerwiegender Grund für eine fristlose Kündigung angesehen werden, da sie das Vertrauensverhältnis zwischen Mieter und Vermieter beeinträchtigt.",
                "§ 280 BGB: Schadensersatzansprüche können auch geltend gemacht werden, wenn durch die falsche Selbstauskunft Schäden entstanden sind, die über die Mietkaution hinausgehen."
                ]
        },
        {
            "input": "Ist der vertragliche Ausschluss von Schadensersatzansprüchen des Mieters gegen den Vermieter wegen Sachschäden, die durch Mängel der Mietsache verursacht sind, für die der Vermieter aufgrund leichter Fahrlässigkeit einzustehen hat, durch die in einem vom Vermieter gestellten Formularmietvertrag über Wohnraum enthaltene Klausel wirksam?", 
            "actual_output": "", 
            "expected_output": "Nein, der vertragliche Ausschluss von Schadensersatzansprüchen des Mieters gegen den Vermieter wegen Sachschäden, die durch Mängel der Mietsache verursacht sind und für die der Vermieter aufgrund leichter Fahrlässigkeit einzustehen hat, ist in einem vom Vermieter gestellten Formularmietvertrag über Wohnraum unwirksam. Nach § 307 Abs. 1 BGB sind Klauseln in Allgemeinen Geschäftsbedingungen unwirksam, wenn sie den Vertragspartner des Verwenders unangemessen benachteiligen. Ein Ausschluss der Haftung des Vermieters für leichte Fahrlässigkeit würde den Mieter unangemessen benachteiligen, da der Vermieter gemäß § 536a Abs. 1 BGB für Schäden einzustehen hat, die durch anfängliche Mängel der Mietsache verursacht werden, wenn diese Mängel durch eine Pflichtverletzung des Vermieters entstehen. Ferner stellt § 309 Nr. 7 BGB klar, dass ein Ausschluss der Haftung für leichte Fahrlässigkeit bei der Verletzung von Leben, Körper oder Gesundheit des Vertragspartners unwirksam ist. Selbst wenn es um Sachschäden geht, würde eine solche Klausel die gesetzlichen Rechte des Mieters erheblich einschränken und ist daher nach der Rechtsprechung des Bundesgerichtshofs (BGH) unwirksam.",
            "retrieval_context": [
                "§ 307 Abs. 1 BGB: Klauseln in Allgemeinen Geschäftsbedingungen sind unwirksam, wenn sie den Vertragspartner des Verwenders unangemessen benachteiligen.",
                "§ 536a Abs. 1 BGB: Der Vermieter ist für Schäden verantwortlich, die durch Mängel der Mietsache verursacht werden, wenn diese Mängel durch eine Pflichtverletzung des Vermieters entstehen.",
                "§ 309 Nr. 7 BGB: Ein Ausschluss der Haftung für leichte Fahrlässigkeit bei der Verletzung von Leben, Körper oder Gesundheit des Vertragspartners ist unwirksam.",
                "BGH-Rechtsprechung: Klauseln in Formularmietverträgen, die den Vermieter von der Haftung für leichte Fahrlässigkeit ausschließen, benachteiligen den Mieter unangemessen und sind daher unwirksam."
                ] 
        },
        {
            "input": "Besteht ein Schadensersatzanspruch des Vermieters, wenn der Mieter die Mietsache bei Rückgabe bunt gestrichen hat?", 
            "actual_output": "", 
            "expected_output": "Ja, der Vermieter kann unter bestimmten Umständen einen Schadensersatzanspruch haben, wenn der Mieter die Mietsache bei Rückgabe bunt gestrichen hat. Nach § 535 Abs. 1 Satz 2 BGB ist der Mieter verpflichtet, die Mietsache in einem vertragsgemäßen Zustand zu erhalten. Wenn die Mieträume in neutralen Farben übergeben wurden und im Mietvertrag festgelegt ist, dass Veränderungen in der Farbgebung bei Auszug rückgängig zu machen sind, hat der Mieter dafür Sorge zu tragen, dass die Räume bei Rückgabe in einem Zustand sind, der eine Weitervermietung ohne weitere Maßnahmen ermöglicht. Der Bundesgerichtshof (BGH) hat in mehreren Urteilen (z.B. BGH, Urteil vom 6. November 2013 – VIII ZR 416/12) entschieden, dass der Mieter verpflichtet ist, die Wände in einer Dekoration zurückzugeben, die dem durchschnittlichen Geschmack entspricht und eine schnelle Weitervermietung ermöglicht. Farbanstriche, die vom neutralen Zustand stark abweichen (z.B. kräftige oder ungewöhnliche Farben), müssen daher vom Mieter vor der Rückgabe beseitigt werden. Unterlässt der Mieter dies, kann der Vermieter Schadensersatz für die Kosten der notwendigen Renovierungsarbeiten verlangen, um die Mietsache wieder in einen vermietbaren Zustand zu versetzen.",
            "retrieval_context": [
                "§ 535 Abs. 1 Satz 2 BGB: Der Mieter ist verpflichtet, die Mietsache in einem vertragsgemäßen Zustand zu erhalten und bei Rückgabe in einem Zustand zu übergeben, der den vertraglichen Vereinbarungen entspricht.",
                "BGH-Urteil vom 6. November 2013 – VIII ZR 416/12: Der Mieter muss die Wohnung bei Auszug in einem Zustand zurückgeben, der dem durchschnittlichen Geschmack entspricht und eine schnelle Weitervermietung ermöglicht. Extreme Farbgestaltungen müssen entfernt werden.",
                "Allgemeine Richtlinien: Farbanstriche, die stark vom neutralen Zustand abweichen, müssen vor der Rückgabe beseitigt werden, um den Zustand für die Wiedervermietung zu optimieren."
                ]
        },
        {
            "input": "Unter welchen Voraussetzungen hat der Vermieter einen Anspruch auf Zustimmung zur Mieterhöhung?", 
            "actual_output": "", 
            "expected_output": "Der Vermieter hat unter bestimmten Voraussetzungen einen Anspruch auf Zustimmung zur Mieterhöhung. Nach § 558 BGB kann der Vermieter vom Mieter die Zustimmung zu einer Erhöhung der Miete bis zur ortsüblichen Vergleichsmiete verlangen, wenn seit der letzten Mieterhöhung mindestens 15 Monate vergangen sind. Die Mieterhöhung darf innerhalb von drei Jahren nicht mehr als 20 Prozent betragen (in angespannten Wohnungsmärkten kann die Kappungsgrenze auf 15 Prozent reduziert sein, gemäß § 558 Abs. 3 BGB). Der Vermieter muss die Mieterhöhung schriftlich begründen, etwa durch Verweis auf einen Mietspiegel, eine Auskunft aus einer Mietdatenbank oder ein Sachverständigengutachten (§ 558a BGB). Der Mieter hat dann eine Überlegungsfrist von zwei Monaten, um der Mieterhöhung zuzustimmen. Tut er dies nicht, kann der Vermieter auf Zustimmung klagen (§ 558b BGB). Der Bundesgerichtshof (BGH) hat in mehreren Urteilen bestätigt, dass der Vermieter diese Zustimmung verlangen kann, wenn die Erhöhung den genannten gesetzlichen Bestimmungen entspricht (z.B. BGH, Urteil vom 13. Oktober 2010 – VIII ZR 26/10).",
            "retrieval_context": [
                "§ 558 BGB: Der Vermieter kann die Zustimmung zur Mieterhöhung verlangen, wenn die Miete bis zur ortsüblichen Vergleichsmiete erhöht werden soll und seit der letzten Mieterhöhung mindestens 15 Monate vergangen sind.",
                "§ 558 Abs. 3 BGB: Die Mieterhöhung darf innerhalb von drei Jahren nicht mehr als 20 Prozent betragen. In angespannten Wohnungsmärkten kann die Kappungsgrenze auf 15 Prozent reduziert sein.",
                "§ 558a BGB: Der Vermieter muss die Mieterhöhung schriftlich begründen, z.B. durch Verweis auf einen Mietspiegel, eine Auskunft aus einer Mietdatenbank oder ein Sachverständigengutachten.",
                "§ 558b BGB: Der Mieter hat eine Überlegungsfrist von zwei Monaten, um der Mieterhöhung zuzustimmen. Wenn der Mieter nicht zustimmt, kann der Vermieter auf Zustimmung klagen.",
                "BGH-Urteil vom 13. Oktober 2010 – VIII ZR 26/10: Der Vermieter kann die Zustimmung zur Mieterhöhung verlangen, wenn die Erhöhung den gesetzlichen Bestimmungen entspricht."
                ] 
        },
        {
            "input": "Unter welchen Voraussetzungen ist eine Klausel zur Renovierungspflicht des Mieters bei Auszug wirksam, und wie stehen diese Pflichten im Verhältnis zu den Rechten des Vermieters?", 
            "actual_output": "", 
            "expected_output": "Eine Klausel zur Renovierungspflicht des Mieters bei Auszug ist nur unter bestimmten Voraussetzungen wirksam. Nach § 307 BGB dürfen Allgemeine Geschäftsbedingungen, zu denen auch formularmäßige Mietverträge zählen, den Mieter nicht unangemessen benachteiligen. Laut der Rechtsprechung des Bundesgerichtshofs (BGH) sind starre Fristenpläne und Klauseln, die den Mieter zur Renovierung unabhängig vom tatsächlichen Zustand der Wohnung verpflichten, unwirksam (BGH, Urteil vom 23. Juni 2004 – VIII ZR 361/03). Renovierungspflichten müssen an den tatsächlichen Zustand der Wohnung anknüpfen. Klauseln, die den Mieter verpflichten, unabhängig von der tatsächlichen Abnutzung regelmäßig zu renovieren oder bei Auszug zu renovieren, sind unwirksam (BGH, Urteil vom 18. März 2015 – VIII ZR 185/14). Ein weiterer wichtiger Aspekt ist die Farbauswahl: Klauseln, die den Mieter verpflichten, bei Auszug in bestimmten Farben zu streichen, sind ebenfalls unwirksam, wenn sie den Gestaltungsspielraum des Mieters während der Mietzeit unangemessen einschränken (BGH, Urteil vom 18. Juni 2008 – VIII ZR 224/07). Der Vermieter hat jedoch das Recht, die Wohnung in einem Zustand zurückzubekommen, der einer ordnungsgemäßen Abnutzung entspricht. Ist die Wohnung übermäßig abgenutzt oder beschädigt, kann der Vermieter Schadensersatzansprüche geltend machen (§ 280 Abs. 1 BGB). Zusammengefasst sind Renovierungsklauseln im Mietvertrag nur dann wirksam, wenn sie den tatsächlichen Zustand der Wohnung berücksichtigen und keine starren Fristen oder Vorgaben zur Farbauswahl enthalten. Andernfalls gelten sie als unangemessene Benachteiligung des Mieters und sind unwirksam.",
            "retrieval_context": [
                "§ 307 BGB: Allgemeine Geschäftsbedingungen sind unwirksam, wenn sie den Vertragspartner unangemessen benachteiligen.",
                "BGH-Urteil vom 23. Juni 2004 – VIII ZR 361/03: Starre Fristenpläne und Klauseln, die den Mieter zur Renovierung unabhängig vom tatsächlichen Zustand der Wohnung verpflichten, sind unwirksam.",
                "BGH-Urteil vom 18. März 2015 – VIII ZR 185/14: Renovierungspflichten müssen an den tatsächlichen Zustand der Wohnung anknüpfen. Regelmäßige Renovierungsverpflichtungen oder solche bei Auszug sind unwirksam, wenn sie nicht den tatsächlichen Zustand berücksichtigen.",
                "BGH-Urteil vom 18. Juni 2008 – VIII ZR 224/07: Klauseln, die den Mieter verpflichten, in bestimmten Farben zu streichen, sind unwirksam, wenn sie den Gestaltungsspielraum des Mieters unangemessen einschränken.",
                "§ 280 Abs. 1 BGB: Schadensersatzansprüche können geltend gemacht werden, wenn die Wohnung übermäßig abgenutzt oder beschädigt ist und dadurch eine Pflichtverletzung des Mieters vorliegt."
                ]
        }
    ]
}

if __name__ == "__main__":

    json.dump(dataset_dict, open("test_dataset.json", "w"), indent=4)
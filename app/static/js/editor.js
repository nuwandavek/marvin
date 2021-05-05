import { displayHeatmap, displayJointHeatmap, setProgress, setSliders, displayModal } from "./display.js";

var attentionViz = 'none';
var styleMode = 'micro-formality';
var editorText = '';

var quillEditor = new Quill('#editor', {
    modules: {
        toolbar: [
            // [{ header: [1, 2, false] }],
            // [{ 'size': ['large', 'huge'] }],
            ['bold', 'italic', 'underline', 'code'],
            [{ 'color': [] }, { 'background': [] }, { 'font': [] }],
        ]
    },
    placeholder: 'Write away ...',
    theme: 'snow'
});


quillEditor.on('text-change', function (delta, oldDelta, source) {
    editorText = quillEditor.getText();
    $("#words").html(editorText.split(/[\w\d\’\'-]+/gi).length - 1);
});


// Init all sliders

const formalitylabels = ['Informal', 'Neutral', 'Formal']
const emolabels = ['Sad', 'Neutral', 'Happy']
const shakelabels = ['Normal', 'Mid', 'High']
const binLabels = ['Wiki', 'Shakespeare', 'Abstract']

$('#formality-slider-micro-formality').slider({
    min: 0,
    max: 2,
    start: 0,
    step: 1,
    interpretLabel: function (value) {
        return formalitylabels[value];
    }
});

$('#formality-slider-micro-joint').slider({
    min: 0,
    max: 2,
    start: 0,
    step: 1,
    interpretLabel: function (value) {
        return formalitylabels[value];
    }
});
$('#emo-slider-micro-joint').slider({
    min: 0,
    max: 2,
    start: 0,
    step: 1,
    interpretLabel: function (value) {
        return emolabels[value];
    }
});

$('#shakespeare-slider-macro-shakespeare').slider({
    min: 0,
    max: 2,
    start: 0,
    step: 1,
    interpretLabel: function (value) {
        return shakelabels[value];
    }
});

$('#binary-slider').slider({
    min: 0,
    max: 2,
    start: 0,
    step: 1,
    interpretLabel: function (value) {
        return binLabels[value];
    }
});
$('input[data-style="none"]').checkbox('set checked');
$('.viz').click((e) => {
    console.log($(e.target).data('style'));
    attentionViz = $(e.target).data('style');
    $('.viz').removeClass('active');
    $(e.target).addClass('active')
});

$('.dropdown').dropdown({
    values: [
        {
            name: 'Micro Styles (Formality)',
            value: 'micro-formality',
            selected: true

        },
        {
            name: 'Micro Styles (Joint)',
            value: 'micro-joint',
        },
        {
            name: 'Macro Styles (Shakespeare)',
            value: 'macro-shakespeare',
        },
        {
            name: 'Macro Styles (Binary)',
            value: 'macro-binary',
        }
    ]
})
    .dropdown({
        onChange: function (value, text, $selectedItem) {
            console.log(value);
            styleMode = value;
            $('.preview-container').hide();
            if (styleMode === "micro-formality") {
                $('#preview-container-micro-formality').show();
            }
            else if (styleMode === "micro-joint") {
                $('#preview-container-micro-joint').show();
            }
            else if (styleMode === "macro-shakespeare") {
                $('#preview-container-macro-shakespeare').show();
            }
            else if (styleMode === "macro-binary") {
                $('#preview-container-macro-binary').show();
            }
        }
    });

$('.analyze').click(() => {
    let txt = editorText;
    let modeSelected = styleMode;
    $.ajax({
        url: 'http://0.0.0.0:5000/analyze',
        crossDomain: true,
        dataType: 'json',
        data: { text: txt, mode: modeSelected },
        success: (d) => {
            // displayHeatmap(d.attn);
            setProgress(d.joint);
            setSliders(d.joint);
            displayJointHeatmap(d.joint, dropdownSelection, quillEditor);

        }
    });
})



$('.transfer').click(() => {
    let txt = editorText;
    let modeSelected = styleMode;
    let controls = {}
    if (styleMode === "micro-formality") {
        controls = {
            formality: $('#formality-slider-micro-formality').slider('get value'),
            suggestions: $('#num-suggestions-micro-formality').val(),
        }
    }
    else if (styleMode === "micro-joint") {
        controls = {
            formality: $('#formality-slider-micro-joint').slider('get value'),
            emo: $('#emo-slider-micro-joint').slider('get value'),
            suggestions: $('#num-suggestions-micro-joint').val(),
        }
    }
    else if (styleMode === "macro-shakespeare") {
        controls = {
            shakespeare: $('#formality-slider-macro-shakespeare').slider('get value'),
        }
    }
    else {
        return 0;
    }


    $.ajax({
        url: 'http://0.0.0.0:5000/transfer',
        crossDomain: true,
        dataType: 'json',
        data: { text: txt, controls: JSON.stringify(controls), mode: modeSelected },
        success: (d) => {
            // console.log(d);
            displayModal(d);
            function selectSuggestion() {
                let k = $(this).data('suggestion-id');
                // console.log(k, d.suggestions[k]);
                quillEditor.setContents([{ insert: d.suggestions[k].text }]);
                $('#transfer-suggestions')
                    .modal('hide');

            }
            $('.suggestion-item').click(selectSuggestion);
            $('#transfer-suggestions')
                .modal({
                    onHide: function () {
                        $('.suggestion-item').unbind('click', selectSuggestion);
                    }
                })
                .modal('show');


        }
    });
})


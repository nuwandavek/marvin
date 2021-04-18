import { displayHeatmap, displayJointHeatmap, setProgress, setSliders, displayModal } from "./display.js";

var dropdownSelection = 'formality';
var editorText = '';

var quillEditor = new Quill('#editor', {
    modules: {
        toolbar: [
            [{ header: [1, 2, false] }],
            ['bold', 'italic', 'underline', 'code'],
            [{ 'color': [] }, { 'background': [] }, { 'font': [] }],
        ]
    },
    placeholder: 'Write away ...',
    theme: 'snow'
});


quillEditor.on('text-change', function (delta, oldDelta, source) {
    editorText = quillEditor.getText();
});

$('#analyze').click(() => {
    let txt = editorText;
    $.ajax({
        url: 'http://0.0.0.0:5000/stats',
        crossDomain: true,
        dataType: 'json',
        data: { text: txt },
        success: (d) => {
            // displayHeatmap(d.attn);
            setProgress(d.joint);
            setSliders(d.joint);
            displayJointHeatmap(d.joint, dropdownSelection, quillEditor);

        }
    });
})


const labels = ['Low', 'Mid', 'High']

$('#formality-slider').slider({
    min: 0,
    max: 2,
    start: 0,
    step: 1,
    interpretLabel: function (value) {
        return labels[value];
    }
});

$('#humor-slider').slider({
    min: 0,
    max: 2,
    start: 0,
    step: 1,
    interpretLabel: function (value) {
        return labels[value];
    }
});

$('.dropdown')
    .dropdown({
        onChange: function (value, text, $selectedItem) {
            console.log(value);
            dropdownSelection = value;
        },
        values: [
            {
                name: 'Formality',
                value: 'formality',
                selected: true

            },
            {
                name: 'Humor',
                value: 'humor',
            }
        ]
    });


$('#transfer').click(() => {
    let txt = editorText;
    let controls = {
        formality: $('#formality-slider').slider('get value'),
        jokes: $('#humor-slider').slider('get value'),
        suggestions: $('#num-suggestions').val(),
    }
    $.ajax({
        url: 'http://0.0.0.0:5000/transfer',
        crossDomain: true,
        dataType: 'json',
        data: { text: txt, controls: JSON.stringify(controls) },
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


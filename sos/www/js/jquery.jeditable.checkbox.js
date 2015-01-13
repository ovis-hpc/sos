// add 'Checkbox' input type to Jeditable.
$.editable.addInputType("checkbox", {
    element: function (settings, original) {
	var hiddenInput = $('<input type="hidden"></input>');
	var div = $('<div class="editable-checkboxes" />');
	if (settings.width != 'none') { div.width(settings.width); }
	if (settings.size) { div.attr('size', settings.size); }
	$(this)
	    .append(div)
	    .append(hiddenInput);
	return (hiddenInput);
    },
    content: function (string, settings, original) {
	function setHiddenInput(justClicked) {
	    var checked = $(':checked', justClicked.parentNode);
	    // combine multiple checked values into one string
	    var checkedString = '';
	    $.each(checked, function (index, val) {
		checkedString += ((checked[index]).value + ", ");
	    });
	    // remove trailing comma from string
	    checkedString = checkedString.slice(0, -2);
	    // set value of hidden input to our string of new data
	    $('input[type=hidden]', justClicked.parentNode.parentNode).val(checkedString);
	}
	var dataList = string;
	// loop to attach checkboxes
	for (var i = 0; i < dataList.length; i++) {
	    var input = $('<input type="checkbox"/>')
		.val(dataList[i]);
	    input.uniqueId();
	    var label = $('<label for="' + input.attr("id") + '"/>');
	    label.append(dataList[i]);
	    var line = $('<br/>');
	    $('div', this)
		.append(input)
		.append(label)
		.append(line);
	    /* bind hidden input to any checkbox click, so that the field is
	     * always current */
	    input.on("click", function () {
		setHiddenInput(this);
	    });
	    /* bind hidden input to mouse focus, so that field is populated even
	     * if no checkboxes are clicked */
	    input.parent().on('focusin', function() {
		setHiddenInput(this);
	    });
	}
	// create array of original selected elements
	var originalData = original.revert.split(',');
	// remove commas and whitespace
	for (var j = 0; j < originalData.length; j++) {
	    originalData[j] = originalData[j].replace(',', '').replace(' ', '');
	}
	// loop again to set selected values
	$('div', this).children().each(function () {
	    for (var k = 0; k < originalData.length; k++) {
		if (originalData.indexOf(this.value) !== -1) {
		    $(this).attr("checked", "checked");
		    break;
		}
	    }
	});
    }
});

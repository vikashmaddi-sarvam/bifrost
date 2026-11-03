package bedrock

import (
	"strings"

	"github.com/maximhq/bifrost/core/schemas"
)

// DeepSeek V3.2 writes tool calls as DSML markup. Bedrock (Converse and Mantle) parses
// them into structured tool calls but leaves the opening tag at the end of the text
// before them; these helpers remove that tag and nothing else.
const (
	dsmlMarker  = "<｜DSML｜"
	dsmlLeak    = dsmlMarker + "function_calls"
	dsmlMaxHeld = len(dsmlLeak) + 16
)

// IsDSMLLeakModel reports whether model is DeepSeek V3.2, the only family that leaks the tag.
func IsDSMLLeakModel(ctx *schemas.BifrostContext, model string) bool {
	return strings.Contains(model, "deepseek.v3.2") || strings.Contains(schemas.ResolveCanonicalModel(ctx, model), "deepseek.v3.2")
}

// StripDSMLLeak removes a trailing leaked DSML opening tag from text.
func StripDSMLLeak(text string) string {
	i := strings.LastIndex(text, dsmlMarker)
	if i < 0 || !isDSMLLeakTail(text[i:]) {
		return text
	}
	if strings.TrimSpace(text[:i]) == "" {
		return ""
	}
	return text[:i]
}

// isDSMLLeakTail reports whether s (starting at the marker) is the leaked tag, or a prefix of it, plus whitespace.
func isDSMLLeakTail(s string) bool {
	return len(s) <= dsmlMaxHeld && strings.HasPrefix(dsmlLeak, strings.TrimRight(s, " \t\r\n"))
}

// StripDSMLLeakChat removes the leaked tag from every choice's message text.
func StripDSMLLeakChat(resp *schemas.BifrostChatResponse) {
	if resp == nil {
		return
	}
	for i := range resp.Choices {
		c := resp.Choices[i].ChatNonStreamResponseChoice
		if c == nil || c.Message == nil || c.Message.Content == nil {
			continue
		}
		if s := c.Message.Content.ContentStr; s != nil {
			if out := StripDSMLLeak(*s); out != *s {
				c.Message.Content.ContentStr = &out
			}
		}
		for j := range c.Message.Content.ContentBlocks {
			if t := c.Message.Content.ContentBlocks[j].Text; t != nil {
				if out := StripDSMLLeak(*t); out != *t {
					c.Message.Content.ContentBlocks[j].Text = &out
				}
			}
		}
	}
}

// stripDSMLLeakConverse removes the leaked tag from a Converse response's text blocks.
func stripDSMLLeakConverse(resp *BedrockConverseResponse) {
	if resp == nil || resp.Output == nil || resp.Output.Message == nil {
		return
	}
	content := resp.Output.Message.Content[:0]
	for _, block := range resp.Output.Message.Content {
		if block.Text != nil {
			out := StripDSMLLeak(*block.Text)
			if out == "" && *block.Text != "" {
				continue // the block held nothing but the tag
			}
			if out != *block.Text {
				block.Text = &out
			}
		}
		content = append(content, block)
	}
	resp.Output.Message.Content = content
}

// DSMLStreamFilter holds back a leaked tag across text deltas of one stream; nil disables it.
type DSMLStreamFilter struct {
	held    string
	holding bool
}

// NewDSMLStreamFilter returns a filter for model, or nil when the model cannot leak the tag.
func NewDSMLStreamFilter(ctx *schemas.BifrostContext, model string) *DSMLStreamFilter {
	if !IsDSMLLeakModel(ctx, model) {
		return nil
	}
	return &DSMLStreamFilter{}
}

// Text filters one text delta and returns the text to emit now.
//
// Text before the marker goes out at once. From the marker on, text is held while it is
// still the leaked tag (at most dsmlMaxHeld bytes) and released whole as soon as it is not.
func (f *DSMLStreamFilter) Text(delta string) string {
	if f == nil {
		return delta
	}
	if f.holding {
		f.held += delta
		if isDSMLLeakTail(f.held) {
			return ""
		}
		out := f.held
		f.held, f.holding = "", false
		return out
	}
	i := strings.Index(delta, dsmlMarker)
	if i < 0 || !isDSMLLeakTail(delta[i:]) {
		return delta
	}
	f.held, f.holding = delta[i:], true
	return delta[:i]
}

// End drops a held tag at a block or message boundary.
func (f *DSMLStreamFilter) End() {
	if f != nil {
		f.held, f.holding = "", false
	}
}

// FilterChatChunk filters an OpenAI-shaped chat stream chunk in place.
func (f *DSMLStreamFilter) FilterChatChunk(resp *schemas.BifrostChatResponse) {
	if f == nil || resp == nil {
		return
	}
	for i := range resp.Choices {
		choice := &resp.Choices[i]
		if d := choice.ChatStreamResponseChoice; d != nil && d.Delta != nil {
			if d.Delta.Content != nil {
				if out := f.Text(*d.Delta.Content); out != *d.Delta.Content {
					if out == "" {
						d.Delta.Content = nil
					} else {
						d.Delta.Content = schemas.Ptr(out)
					}
				}
			}
			if len(d.Delta.ToolCalls) > 0 {
				f.End()
			}
		}
		if choice.FinishReason != nil {
			f.End()
		}
	}
}

// filterConverseEvent filters a Converse stream event in place.
//
// Any event other than a text delta (block stop, tool start, message stop) ends the text block.
func (f *DSMLStreamFilter) filterConverseEvent(ev *BedrockStreamEvent) {
	if f == nil {
		return
	}
	if ev.Delta == nil || ev.Delta.Text == nil {
		f.End()
		return
	}
	if out := f.Text(*ev.Delta.Text); out != *ev.Delta.Text {
		ev.Delta.Text = schemas.Ptr(out)
	}
}
